/*
 * Copyright 2010-2021 JetBrains s.r.o. and Kotlin Programming Language contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package org.jetbrains.kotlin.fir.analysis.diagnostics

import com.intellij.openapi.util.TextRange
import com.intellij.psi.PsiElement
import org.jetbrains.kotlin.cfg.UnreachableCode
import org.jetbrains.kotlin.diagnostics.DiagnosticMarker
import org.jetbrains.kotlin.diagnostics.PositioningStrategy
import org.jetbrains.kotlin.fir.FirPsiSourceElement
import org.jetbrains.kotlin.fir.FirSourceElement
import org.jetbrains.kotlin.fir.psi
import org.jetbrains.kotlin.psi.KtElement
import org.jetbrains.kotlin.psi.KtNamedDeclaration


object FirPsiPositioningStrategies {
    val UNREACHABLE_CODE = object : PositioningStrategy<PsiElement>() {
        override fun markDiagnostic(diagnostic: DiagnosticMarker): List<TextRange> {
            //todo it is better to implement arguments extraction in FirDiagnosticFactory, but kotlin struggle with checking types in it atm
            @Suppress("UNCHECKED_CAST")
            val typed = diagnostic as FirDiagnosticWithParameters2<Set<FirSourceElement>, Set<FirSourceElement>>
            val source = diagnostic.element as FirPsiSourceElement
            return UnreachableCode.getUnreachableTextRanges(
                source.psi as KtElement,
                typed.a.mapNotNull { it.psi as? KtElement }.toSet(),
                typed.b.mapNotNull { it.psi as? KtElement }.toSet()
            )
        }
    }

    val ACTUAL_DECLARATION_NAME = object : PositioningStrategy<PsiElement>() {
        override fun markDiagnostic(diagnostic: DiagnosticMarker): List<TextRange> {
            require(diagnostic is FirDiagnostic)
            val element = diagnostic.element.psi ?: return emptyList()
            (element as? KtNamedDeclaration)?.nameIdentifier?.let { nameIdentifier ->
                return mark(nameIdentifier)
            }
            return mark(element)
        }
    }
}
