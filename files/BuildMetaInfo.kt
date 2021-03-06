/*
 * Copyright 2010-2018 JetBrains s.r.o. and Kotlin Programming Language contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package org.jetbrains.kotlin.build

import org.jetbrains.kotlin.cli.common.arguments.CommonCompilerArguments
import org.jetbrains.kotlin.config.KotlinCompilerVersion
import org.jetbrains.kotlin.config.LanguageVersion
import org.jetbrains.kotlin.metadata.deserialization.BinaryVersion
import kotlin.reflect.KClass

interface BuildMetaInfo {
    val isEAP: Boolean
    val compilerBuildVersion: String
    val languageVersionString: String
    val apiVersionString: String
    val multiplatformEnable: Boolean
    val metadataVersionMajor: Int
    val metadataVersionMinor: Int
    val metadataVersionPatch: Int
    val ownVersion: Int
    val coroutinesVersion: Int
    val multiplatformVersion: Int
}

abstract class BuildMetaInfoFactory<T : BuildMetaInfo>(private val metaInfoClass: KClass<T>) {
    protected abstract fun create(
        isEAP: Boolean,
        compilerBuildVersion: String,
        languageVersionString: String,
        apiVersionString: String,
        multiplatformEnable: Boolean,
        ownVersion: Int,
        coroutinesVersion: Int,
        multiplatformVersion: Int,
        metadataVersionArray: IntArray?
    ): T

    fun create(args: CommonCompilerArguments): T {
        val languageVersion = args.languageVersion?.let { LanguageVersion.fromVersionString(it) } ?: LanguageVersion.LATEST_STABLE

        return create(
            isEAP = !languageVersion.isStable,
            compilerBuildVersion = KotlinCompilerVersion.VERSION,
            languageVersionString = languageVersion.versionString,
            apiVersionString = args.apiVersion ?: languageVersion.versionString,
            multiplatformEnable = args.multiPlatform,
            ownVersion = OWN_VERSION,
            coroutinesVersion = COROUTINES_VERSION,
            multiplatformVersion = MULTIPLATFORM_VERSION,
            metadataVersionArray = args.metadataVersion?.let { BinaryVersion.parseVersionArray(it) }
        )
    }

    fun serializeToString(args: CommonCompilerArguments): String =
        serializeToString(create(args))

    fun serializeToString(info: T): String =
        serializeToPlainText(info, metaInfoClass)

    fun deserializeFromString(str: String): T? =
        deserializeFromPlainText(str, metaInfoClass)

    companion object {
        const val OWN_VERSION: Int = 0
        const val COROUTINES_VERSION: Int = 0
        const val MULTIPLATFORM_VERSION: Int = 0
    }
}
