/*
 * Copyright 2010-2020 JetBrains s.r.o. and Kotlin Programming Language contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package org.jetbrains.kotlin.fir.analysis.diagnostics

import com.intellij.openapi.util.TextRange
import com.intellij.psi.PsiElement
import org.jetbrains.kotlin.diagnostics.PositioningStrategy
import org.jetbrains.kotlin.fir.FirLightSourceElement
import org.jetbrains.kotlin.fir.FirPsiSourceElement
import org.jetbrains.kotlin.fir.FirSourceElement
import org.jetbrains.kotlin.fir.psi

class SourceElementPositioningStrategy(
    private val lightTreeStrategy: LightTreePositioningStrategy,
    private val psiStrategy: PositioningStrategy<*>
) {
    fun markDiagnostic(diagnostic: FirDiagnostic): List<TextRange> {
        return when (val element = diagnostic.element) {
            is FirPsiSourceElement -> psiStrategy.markDiagnostic(diagnostic)
            is FirLightSourceElement -> lightTreeStrategy.markFirDiagnostic(element, diagnostic)
        }
    }

    fun isValid(element: FirSourceElement): Boolean {
        return when (element) {
            is FirPsiSourceElement -> psiStrategy.hackyIsValid(element.psi)
            is FirLightSourceElement -> lightTreeStrategy.isValid(element.lighterASTNode, element.treeStructure)
        }
    }

    private fun PositioningStrategy<*>.hackyIsValid(psi: PsiElement): Boolean {
        @Suppress("UNCHECKED_CAST")
        return (this as PositioningStrategy<PsiElement>).isValid(psi)
    }

    companion object {
        val DEFAULT: SourceElementPositioningStrategy = SourceElementPositioningStrategies.DEFAULT
    }
}
