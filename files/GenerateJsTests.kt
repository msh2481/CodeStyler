/*
 * Copyright 2010-2020 JetBrains s.r.o. and Kotlin Programming Language contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package org.jetbrains.kotlin.generators.tests

import org.jetbrains.kotlin.generators.impl.generateTestGroupSuite
import org.jetbrains.kotlin.js.test.AbstractDceTest
import org.jetbrains.kotlin.js.test.AbstractJsLineNumberTest
import org.jetbrains.kotlin.js.test.compatibility.binary.AbstractJsKlibBinaryCompatibilityTest
import org.jetbrains.kotlin.js.test.es6.semantics.AbstractIrBoxJsES6Test
import org.jetbrains.kotlin.js.test.es6.semantics.AbstractIrJsCodegenBoxES6Test
import org.jetbrains.kotlin.js.test.es6.semantics.AbstractIrJsCodegenInlineES6Test
import org.jetbrains.kotlin.js.test.es6.semantics.AbstractIrJsTypeScriptExportES6Test
import org.jetbrains.kotlin.js.test.ir.semantics.*
import org.jetbrains.kotlin.js.test.semantics.*
import org.jetbrains.kotlin.js.test.wasm.semantics.AbstractIrCodegenBoxWasmTest
import org.jetbrains.kotlin.js.test.wasm.semantics.AbstractIrCodegenWasmJsInteropWasmTest
import org.jetbrains.kotlin.js.test.wasm.semantics.AbstractJsTranslatorWasmTest
import org.jetbrains.kotlin.test.TargetBackend

fun main(args: Array<String>) {
    System.setProperty("java.awt.headless", "true")

    // TODO: repair these tests
    //generateTestDataForReservedWords()

    generateTestGroupSuite(args) {
        testGroup("js/js.tests/tests-gen", "js/js.translator/testData", testRunnerMethodName = "runTest0") {
            testClass<AbstractBoxJsTest> {
                model("box/", pattern = "^([^_](.+))\\.kt$", targetBackend = TargetBackend.JS)
            }

            testClass<AbstractIrBoxJsTest> {
                model("box/", pattern = "^([^_](.+))\\.kt$", targetBackend = TargetBackend.JS_IR)
            }

            testClass<AbstractJsTranslatorWasmTest> {
                model("box/main", pattern = "^([^_](.+))\\.kt$", targetBackend = TargetBackend.WASM)
                model("box/kotlin.test/", pattern = "^([^_](.+))\\.kt$", targetBackend = TargetBackend.WASM)
            }

            testClass<AbstractIrBoxJsES6Test> {
                model("box/", pattern = "^([^_](.+))\\.kt$", targetBackend = TargetBackend.JS_IR_ES6)
            }

            testClass<AbstractIrJsTypeScriptExportTest> {
                model("typescript-export/", pattern = "^([^_](.+))\\.kt$", targetBackend = TargetBackend.JS_IR)
            }

            testClass<AbstractIrJsTypeScriptExportES6Test> {
                model("typescript-export/", pattern = "^([^_](.+))\\.kt$", targetBackend = TargetBackend.JS_IR_ES6)
            }

            testClass<AbstractSourceMapGenerationSmokeTest> {
                model("sourcemap/", pattern = "^([^_](.+))\\.kt$", targetBackend = TargetBackend.JS)
            }

            testClass<AbstractOutputPrefixPostfixTest> {
                model("outputPrefixPostfix/", pattern = "^([^_](.+))\\.kt$", targetBackend = TargetBackend.JS)
            }

            testClass<AbstractDceTest> {
                model("dce/", pattern = "(.+)\\.js", targetBackend = TargetBackend.JS)
            }

            testClass<AbstractJsLineNumberTest> {
                model("lineNumbers/", pattern = "^([^_](.+))\\.kt$", targetBackend = TargetBackend.JS)
            }
        }

        testGroup("js/js.tests/tests-gen", "compiler/testData", testRunnerMethodName = "runTest0") {
            val jvmOnlyBoxTests = listOf(
                "testsWithJava9",
                "testsWithJava15",
                "testsWithJava17",
            )

            testClass<AbstractJsCodegenBoxTest> {
                model("codegen/box", targetBackend = TargetBackend.JS, excludeDirs = jvmOnlyBoxTests + "compileKotlinAgainstKotlin")
            }

            testClass<AbstractIrJsCodegenBoxTest> {
                model("codegen/box", targetBackend = TargetBackend.JS_IR, excludeDirs = jvmOnlyBoxTests + "compileKotlinAgainstKotlin")
            }

            testClass<AbstractIrJsCodegenBoxErrorTest> {
                model("codegen/boxError", targetBackend = TargetBackend.JS_IR, excludeDirs = jvmOnlyBoxTests + "compileKotlinAgainstKotlin")
            }

            testClass<AbstractIrCodegenBoxWasmTest> {
                model(
                    "codegen/box", pattern = "^([^_](.+))\\.kt$", targetBackend = TargetBackend.WASM, excludeDirs = listOf(
                        
                        // TODO: Support reflection
                        "toArray", "classLiteral", "reflection",

                        // TODO: Add stdlib
                        "contracts", "platformTypes",

                        // TODO: ArrayList
                        "ranges/stepped/unsigned",

                        // TODO: Support delegated properties
                        "delegatedProperty",

                        "compileKotlinAgainstKotlin"
                    ) + jvmOnlyBoxTests
                )
            }

            testClass<AbstractIrCodegenWasmJsInteropWasmTest> {
                model("codegen/boxWasmJsInterop", targetBackend = TargetBackend.WASM)
            }

            testClass<AbstractIrCodegenWasmJsInteropJsTest> {
                model("codegen/boxWasmJsInterop", targetBackend = TargetBackend.JS_IR)
            }

            testClass<AbstractIrJsCodegenBoxES6Test> {
                model("codegen/box", targetBackend = TargetBackend.JS_IR_ES6, excludeDirs = jvmOnlyBoxTests)
            }

            testClass<AbstractJsCodegenInlineTest> {
                model("codegen/boxInline/", targetBackend = TargetBackend.JS)
            }

            testClass<AbstractIrJsCodegenInlineTest> {
                model("codegen/boxInline/", targetBackend = TargetBackend.JS_IR)
            }

            testClass<AbstractIrJsCodegenInlineES6Test> {
                model("codegen/boxInline/", targetBackend = TargetBackend.JS_IR_ES6)
            }

            testClass<AbstractJsLegacyPrimitiveArraysBoxTest> {
                model("codegen/box/arrays", targetBackend = TargetBackend.JS)
            }
        }

        testGroup("js/js.tests/tests-gen", "compiler/testData/binaryCompatibility", testRunnerMethodName = "runTest0") {
            testClass<AbstractJsKlibBinaryCompatibilityTest> {
                model("klibEvolution", targetBackend = TargetBackend.JS_IR)
            }
        }
    }
}
