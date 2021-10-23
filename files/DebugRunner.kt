/*
 * Copyright 2010-2021 JetBrains s.r.o. and Kotlin Programming Language contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package org.jetbrains.kotlin.test.backend.handlers

import com.sun.jdi.*
import com.sun.jdi.event.*
import com.sun.jdi.request.EventRequest.SUSPEND_ALL
import com.sun.jdi.request.StepRequest
import com.sun.tools.jdi.SocketAttachingConnector
import org.jetbrains.kotlin.test.TargetBackend
import org.jetbrains.kotlin.test.model.TestModule
import org.jetbrains.kotlin.test.services.JUnit5Assertions.assertEqualsToFile
import org.jetbrains.kotlin.test.services.TestServices
import org.jetbrains.kotlin.test.services.sourceProviders.MainFunctionForBlackBoxTestsSourceProvider.Companion.BOX_MAIN_FILE_NAME
import java.io.File
import java.net.URL

open class LoggedData(val line: Int, val isSynthetic: Boolean, val expectation: String)

abstract class DebugRunner(testServices: TestServices) : JvmBoxRunner(testServices) {

    companion object {
        const val EXPECTATIONS_MARKER = "// EXPECTATIONS"
        const val FORCE_STEP_INTO_MARKER = "// FORCE_STEP_INTO"
        const val JVM_EXPECTATIONS_MARKER = "$EXPECTATIONS_MARKER JVM"
        const val JVM_IR_EXPECTATIONS_MARKER = "$EXPECTATIONS_MARKER JVM_IR"

        val BOX_MAIN_FILE_CLASS_NAME = BOX_MAIN_FILE_NAME.replace(".kt", "Kt")
    }

    private var wholeFile = File("")
    private var backend = TargetBackend.JVM

    abstract fun storeStep(loggedItems: ArrayList<LoggedData>, event: Event)

    override fun launchSeparateJvmProcess(
        javaExe: File,
        module: TestModule,
        classPath: List<URL>,
        mainClassAndArguments: List<String>
    ): Process {
        // Extract target backend and the full test file used to extract test expectations.
        backend = module.targetBackend ?: backend
        wholeFile = module.files.single { it.name == "test.kt" }.originalFile

        // Setup the java process to suspend waiting for debugging connection on a free port.
        val command = listOfNotNull(
            javaExe.absolutePath,
            "-agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=127.0.0.1:0",
            "-ea",
            "-classpath",
            classPath.joinToString(File.pathSeparator, transform = { File(it.toURI()).absolutePath }),
        ) + mainClassAndArguments

        val process = ProcessBuilder(command).start()

        // Extract the chosen port from the output of the newly started java process.
        // The java process prints a line with the format:
        //
        //      Listening for transport dt_socket at address: <port number>
        val port = process.inputStream.bufferedReader().readLine().split("address:").last().trim().toInt()

        // Attach debugger to the separate java process, setup initial event requests,
        // and run the debugger loop to step through the program.
        val virtualMachine = attachDebugger(port)
        setupMethodEntryAndExitRequests(virtualMachine)
        runDebugEventLoop(virtualMachine)

        return process
    }

    // Debug event loop to step through a test program.
    private fun runDebugEventLoop(virtualMachine: VirtualMachine) {
        val manager = virtualMachine.eventRequestManager()
        val loggedItems = ArrayList<LoggedData>()
        var inBoxMethod = false
        vmLoop@
        while (true) {
            val eventSet = virtualMachine.eventQueue().remove(1000) ?: continue
            for (event in eventSet) {
                when (event) {
                    is VMDeathEvent, is VMDisconnectEvent -> {
                        break@vmLoop
                    }
                    // We start VM with option 'suspend=n', in case VMStartEvent is still received, discard.
                    is VMStartEvent -> {

                    }
                    is MethodEntryEvent -> {
                        if (!inBoxMethod && event.location().method().name() == "box") {
                            if (manager.stepRequests().isEmpty()) {
                                // Create line stepping request to get all normal line steps starting now.
                                val stepReq = manager.createStepRequest(event.thread(), StepRequest.STEP_LINE, StepRequest.STEP_INTO)
                                stepReq.setSuspendPolicy(SUSPEND_ALL)
                                stepReq.addClassExclusionFilter("java.*")
                                stepReq.addClassExclusionFilter("sun.*")
                                stepReq.addClassExclusionFilter("kotlin.*")
                                // Create class prepare request to be able to set breakpoints on class initializer lines.
                                // There are no line stepping events for class initializers, so we depend on breakpoints.
                                val prepareReq = manager.createClassPrepareRequest()
                                prepareReq.setSuspendPolicy(SUSPEND_ALL)
                                prepareReq.addClassExclusionFilter("java.*")
                                prepareReq.addClassExclusionFilter("sun.*")
                                prepareReq.addClassExclusionFilter("kotlin.*")
                            }
                            manager.stepRequests().map { it.enable() }
                            manager.classPrepareRequests().map { it.enable() }
                            inBoxMethod = true
                            storeStep(loggedItems, event)
                        }
                    }
                    is StepEvent -> {
                        if (inBoxMethod) {
                            // Handle the case where an Exception causing program to exit without MethodExitEvent.
                            if (event.location().method().name() == "main" &&
                                event.location().declaringType().name().contains(BOX_MAIN_FILE_CLASS_NAME)
                            ) {
                                manager.stepRequests().map { it.disable() }
                                manager.classPrepareRequests().map { it.disable() }
                                manager.breakpointRequests().map { it.disable() }
                                break@vmLoop
                            }
                            storeStep(loggedItems, event)
                        }
                    }
                    is MethodExitEvent -> {
                        if (event.location().method().name() == "box") {
                            manager.stepRequests().map { it.disable() }
                            manager.classPrepareRequests().map { it.disable() }
                            manager.breakpointRequests().map { it.disable() }
                            break@vmLoop
                        }
                    }
                    is ClassPrepareEvent -> {
                        if (inBoxMethod) {
                            val initializer = event.referenceType().methods().find { it.isStaticInitializer }
                            try {
                                initializer?.allLineLocations()?.forEach {
                                    manager.createBreakpointRequest(it).enable()
                                }
                            } catch (e: AbsentInformationException) {
                                // If there is no line information, do not set breakpoints.
                            }
                        }
                    }
                    is BreakpointEvent -> {
                        if (inBoxMethod) {
                            storeStep(loggedItems, event)
                        }
                    }
                    else -> {
                        throw IllegalStateException("event not handled: $event")
                    }
                }
            }
            eventSet.resume()
        }
        checkResult(wholeFile, loggedItems)
        virtualMachine.resume()
    }

    fun Location.formatAsExpectation(): String {
        val synthetic = if (method().isSynthetic) " (synthetic)" else ""
        return "${sourceName()}:${lineNumber()} ${method().name()}$synthetic"
    }

    fun checkResult(wholeFile: File, loggedItems: List<LoggedData>) {
        val actual = mutableListOf<String>()
        val lines = wholeFile.readLines()
        val forceStepInto = lines.any { it.startsWith(FORCE_STEP_INTO_MARKER) }

        val actualLineNumbers = compressSequencesWithoutLinenumber(loggedItems)
            .filter {
                // Ignore synthetic code with no line number information unless force step into behavior is requested.
                forceStepInto || !it.isSynthetic
            }
            .map { "// ${it.expectation}" }
        val actualLineNumbersIterator = actualLineNumbers.iterator()

        val lineIterator = lines.iterator()
        for (line in lineIterator) {
            actual.add(line)
            if (line.startsWith(EXPECTATIONS_MARKER) || line.startsWith(FORCE_STEP_INTO_MARKER)) break
        }

        var currentBackend = TargetBackend.ANY
        for (line in lineIterator) {
            if (line.isEmpty()) {
                actual.add(line)
                continue
            }
            if (line.startsWith(EXPECTATIONS_MARKER)) {
                actual.add(line)
                currentBackend = when (line) {
                    EXPECTATIONS_MARKER -> TargetBackend.ANY
                    JVM_EXPECTATIONS_MARKER -> TargetBackend.JVM
                    JVM_IR_EXPECTATIONS_MARKER -> TargetBackend.JVM_IR
                    else -> error("Expected JVM backend: $line")
                }
                continue
            }
            if (currentBackend == TargetBackend.ANY || currentBackend == backend) {
                if (actualLineNumbersIterator.hasNext()) {
                    actual.add(actualLineNumbersIterator.next())
                }
            } else {
                actual.add(line)
            }
        }

        actualLineNumbersIterator.forEach { actual.add(it) }

        assertEqualsToFile(wholeFile, actual.joinToString("\n"))
    }

    // Compresses sequences of the same location without line number in the log:
    // specifically removes locations without linenumber, that would otherwise
    // print as byte offsets. This avoids overspecifying code generation
    // strategy in debug tests.
    fun compressSequencesWithoutLinenumber(loggedItems: List<LoggedData>): List<LoggedData> {
        if (loggedItems.isEmpty()) return listOf()

        val logIterator = loggedItems.iterator()
        var currentItem = logIterator.next()
        val result = mutableListOf(currentItem)

        for (logItem in logIterator) {
            if (currentItem.line != -1 || currentItem.expectation != logItem.expectation) {
                result.add(logItem)
                currentItem = logItem
            }
        }

        return result
    }

    fun setupMethodEntryAndExitRequests(virtualMachine: VirtualMachine) {
        val manager = virtualMachine.eventRequestManager()

        val methodEntryReq = manager.createMethodEntryRequest()
        methodEntryReq.addClassFilter("TestKt")
        methodEntryReq.setSuspendPolicy(SUSPEND_ALL)
        methodEntryReq.enable()

        val methodExitReq = manager.createMethodExitRequest()
        methodExitReq.addClassFilter("TestKt")
        methodExitReq.setSuspendPolicy(SUSPEND_ALL)
        methodExitReq.enable()
    }

    private fun attachDebugger(port: Int): VirtualMachine {
        val connector = SocketAttachingConnector()
        val virtualMachine = connector.attach(connector.defaultArguments().toMutableMap().apply {
            getValue("port").setValue("$port")
            getValue("hostname").setValue("127.0.0.1")
        })
        return virtualMachine
    }

}

class SteppingDebugRunner(testServices: TestServices) : DebugRunner(testServices) {
    override fun storeStep(loggedItems: ArrayList<LoggedData>, event: Event) {
        assert(event is LocatableEvent)
        val location = (event as LocatableEvent).location()
        loggedItems.add(
            LoggedData(
                location.lineNumber(),
                location.method().isSynthetic,
                location.formatAsExpectation()
            )
        )
    }
}

class LocalVariableDebugRunner(testServices: TestServices) : DebugRunner(testServices) {
    interface LocalValue

    class LocalPrimitive(val value: String, val valueType: String) : LocalValue {
        override fun toString(): String {
            return "$value:$valueType"
        }
    }

    class LocalReference(val id: String, val referenceType: String) : LocalValue {
        override fun toString(): String {
            return referenceType
        }
    }

    class LocalNullValue : LocalValue {
        override fun toString(): String {
            return "null"
        }
    }

    class LocalVariableRecord(
        val variable: String,
        val variableType: String,
        val value: LocalValue
    ) {
        override fun toString(): String {
            return "$variable:$variableType=$value"
        }
    }

    private fun toRecord(frame: StackFrame, variable: LocalVariable): LocalVariableRecord {
        val value = frame.getValue(variable)
        val valueRecord = if (value == null) {
            LocalNullValue()
        } else if (value is ObjectReference && value.referenceType().name() != "java.lang.String") {
            LocalReference(value.uniqueID().toString(), value.referenceType().name())
        } else {
            LocalPrimitive(value.toString(), value.type().name())
        }
        return LocalVariableRecord(variable.name(), variable.typeName(), valueRecord)
    }

    private fun waitUntil(condition: () -> Boolean) {
        while (!condition()) {
            Thread.sleep(10)
        }
    }

    override fun storeStep(loggedItems: ArrayList<LoggedData>, event: Event) {
        val locatableEvent = event as LocatableEvent
        waitUntil { locatableEvent.thread().isSuspended }
        val location = locatableEvent.location()
        if (location.method().isSynthetic) return

        val frame = locatableEvent.thread().frame(0)
        val visibleVars = try {
            frame.visibleVariables().map { variable -> toRecord(frame, variable) }
        } catch (e: AbsentInformationException) {
            // Local variable table completely absent - not distinguished from an empty table.
            listOf()
        }
        loggedItems.add(
            LoggedData(
                location.lineNumber(),
                false,
                "${location.formatAsExpectation()}: ${visibleVars.joinToString(", ")}".trim()
            )
        )
    }
}