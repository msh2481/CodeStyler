/*
 * Copyright 2010-2020 JetBrains s.r.o. and Kotlin Programming Language contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package org.jetbrains.kotlin.fir.java

import org.jetbrains.kotlin.descriptors.Visibilities
import org.jetbrains.kotlin.descriptors.Visibility
import org.jetbrains.kotlin.descriptors.java.JavaVisibilities
import org.jetbrains.kotlin.fir.*
import org.jetbrains.kotlin.fir.declarations.FirDeclaration
import org.jetbrains.kotlin.fir.declarations.FirFile
import org.jetbrains.kotlin.fir.declarations.synthetic.FirSyntheticPropertyAccessor
import org.jetbrains.kotlin.fir.declarations.utils.visibility
import org.jetbrains.kotlin.fir.resolve.calls.FirSimpleSyntheticPropertySymbol
import org.jetbrains.kotlin.fir.resolve.calls.ReceiverValue
import org.jetbrains.kotlin.fir.symbols.FirBasedSymbol
import org.jetbrains.kotlin.fir.symbols.impl.FirNamedFunctionSymbol
import org.jetbrains.kotlin.fir.symbols.impl.FirPropertyAccessorSymbol
import org.jetbrains.kotlin.fir.symbols.impl.FirVariableSymbol

@NoMutableState
object FirJavaVisibilityChecker : FirVisibilityChecker() {
    override fun platformVisibilityCheck(
        declarationVisibility: Visibility,
        symbol: FirBasedSymbol<*>,
        useSiteFile: FirFile,
        containingDeclarations: List<FirDeclaration>,
        dispatchReceiver: ReceiverValue?,
        session: FirSession,
        isCallToPropertySetter: Boolean,
    ): Boolean {
        return when (declarationVisibility) {
            JavaVisibilities.ProtectedAndPackage, JavaVisibilities.ProtectedStaticVisibility -> {
                if (symbol.packageFqName() == useSiteFile.packageFqName) {
                    true
                } else {
                    val ownerLookupTag = symbol.getOwnerLookupTag() ?: return false
                    if (canSeeProtectedMemberOf(
                            containingDeclarations, dispatchReceiver, ownerLookupTag, session,
                            isVariableOrNamedFunction = symbol is FirVariableSymbol || symbol is FirNamedFunctionSymbol || symbol is FirPropertyAccessorSymbol,
                            isSyntheticProperty = symbol.fir is FirSyntheticPropertyAccessor
                        )
                    ) return true

                    // FE1.0 allows calling public setters with property assignment syntax if the getter is protected.
                    if (!isCallToPropertySetter || symbol !is FirSimpleSyntheticPropertySymbol) return false
                    symbol.setterSymbol?.visibility == Visibilities.Public
                }
            }

            JavaVisibilities.PackageVisibility -> {
                if (symbol.packageFqName() == useSiteFile.packageFqName) {
                    true
                } else if (symbol.fir is FirSyntheticPropertyAccessor) {
                    symbol.getOwnerLookupTag()?.classId?.packageFqName == useSiteFile.packageFqName
                } else {
                    false
                }
            }

            else -> true
        }
    }
}
