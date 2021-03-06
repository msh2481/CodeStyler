/*
 * Copyright 2010-2020 JetBrains s.r.o. and Kotlin Programming Language contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package org.jetbrains.kotlin.fir.analysis.diagnostics

import com.intellij.psi.PsiElement
import org.jetbrains.kotlin.config.LanguageFeature
import org.jetbrains.kotlin.diagnostics.Severity
import kotlin.properties.ReadOnlyProperty
import kotlin.reflect.KClass
import kotlin.reflect.KProperty

inline fun <reified P : PsiElement> warning0(
    positioningStrategy: SourceElementPositioningStrategy = SourceElementPositioningStrategy.DEFAULT
): DiagnosticFactory0DelegateProvider {
    return DiagnosticFactory0DelegateProvider(Severity.WARNING, positioningStrategy, P::class)
}

inline fun <reified P : PsiElement, A> warning1(
    positioningStrategy: SourceElementPositioningStrategy = SourceElementPositioningStrategy.DEFAULT
): DiagnosticFactory1DelegateProvider<A> {
    return DiagnosticFactory1DelegateProvider(Severity.WARNING, positioningStrategy, P::class)
}

inline fun <reified P : PsiElement, A, B> warning2(
    positioningStrategy: SourceElementPositioningStrategy = SourceElementPositioningStrategy.DEFAULT
): DiagnosticFactory2DelegateProvider<A, B> {
    return DiagnosticFactory2DelegateProvider(Severity.WARNING, positioningStrategy, P::class)
}

inline fun <reified P : PsiElement, A, B, C> warning3(
    positioningStrategy: SourceElementPositioningStrategy = SourceElementPositioningStrategy.DEFAULT
): DiagnosticFactory3DelegateProvider<A, B, C> {
    return DiagnosticFactory3DelegateProvider(Severity.WARNING, positioningStrategy, P::class)
}

inline fun <reified P : PsiElement, A, B, C, D> warning4(
    positioningStrategy: SourceElementPositioningStrategy = SourceElementPositioningStrategy.DEFAULT
): DiagnosticFactory4DelegateProvider<A, B, C, D> {
    return DiagnosticFactory4DelegateProvider(Severity.WARNING, positioningStrategy, P::class)
}

inline fun <reified P : PsiElement> error0(
    positioningStrategy: SourceElementPositioningStrategy = SourceElementPositioningStrategy.DEFAULT
): DiagnosticFactory0DelegateProvider {
    return DiagnosticFactory0DelegateProvider(Severity.ERROR, positioningStrategy, P::class)
}

inline fun <reified P : PsiElement, A> error1(
    positioningStrategy: SourceElementPositioningStrategy = SourceElementPositioningStrategy.DEFAULT
): DiagnosticFactory1DelegateProvider<A> {
    return DiagnosticFactory1DelegateProvider(Severity.ERROR, positioningStrategy, P::class)
}

inline fun <reified P : PsiElement, A, B> error2(
    positioningStrategy: SourceElementPositioningStrategy = SourceElementPositioningStrategy.DEFAULT
): DiagnosticFactory2DelegateProvider<A, B> {
    return DiagnosticFactory2DelegateProvider(Severity.ERROR, positioningStrategy, P::class)
}

inline fun <reified P : PsiElement, A, B, C> error3(
    positioningStrategy: SourceElementPositioningStrategy = SourceElementPositioningStrategy.DEFAULT
): DiagnosticFactory3DelegateProvider<A, B, C> {
    return DiagnosticFactory3DelegateProvider(Severity.ERROR, positioningStrategy, P::class)
}

inline fun <reified P : PsiElement, A, B, C, D> error4(
    positioningStrategy: SourceElementPositioningStrategy = SourceElementPositioningStrategy.DEFAULT
): DiagnosticFactory4DelegateProvider<A, B, C, D> {
    return DiagnosticFactory4DelegateProvider(Severity.ERROR, positioningStrategy, P::class)
}

inline fun <reified P : PsiElement> deprecationError0(
    featureForError: LanguageFeature,
    positioningStrategy: SourceElementPositioningStrategy = SourceElementPositioningStrategy.DEFAULT
): DeprecationDiagnosticFactory0DelegateProvider {
    return DeprecationDiagnosticFactory0DelegateProvider(featureForError, positioningStrategy, P::class)
}

inline fun <reified P : PsiElement, A> deprecationError1(
    featureForError: LanguageFeature,
    positioningStrategy: SourceElementPositioningStrategy = SourceElementPositioningStrategy.DEFAULT
): DeprecationDiagnosticFactory1DelegateProvider<A> {
    return DeprecationDiagnosticFactory1DelegateProvider(featureForError, positioningStrategy, P::class)
}

inline fun <reified P : PsiElement, A, B> deprecationError2(
    featureForError: LanguageFeature,
    positioningStrategy: SourceElementPositioningStrategy = SourceElementPositioningStrategy.DEFAULT
): DeprecationDiagnosticFactory2DelegateProvider<A, B> {
    return DeprecationDiagnosticFactory2DelegateProvider(featureForError, positioningStrategy, P::class)
}

inline fun <reified P : PsiElement, A, B, C> deprecationError3(
    featureForError: LanguageFeature,
    positioningStrategy: SourceElementPositioningStrategy = SourceElementPositioningStrategy.DEFAULT
): DeprecationDiagnosticFactory3DelegateProvider<A, B, C> {
    return DeprecationDiagnosticFactory3DelegateProvider(featureForError, positioningStrategy, P::class)
}

inline fun <reified P : PsiElement, A, B, C, D> deprecationError4(
    featureForError: LanguageFeature,
    positioningStrategy: SourceElementPositioningStrategy = SourceElementPositioningStrategy.DEFAULT
): DeprecationDiagnosticFactory4DelegateProvider<A, B, C, D> {
    return DeprecationDiagnosticFactory4DelegateProvider(featureForError, positioningStrategy, P::class)
}

// ------------------------------ Providers ------------------------------

class DiagnosticFactory0DelegateProvider(
    private val severity: Severity,
    private val positioningStrategy: SourceElementPositioningStrategy,
    private val psiType: KClass<*>
) {
    operator fun provideDelegate(thisRef: Any?, prop: KProperty<*>): ReadOnlyProperty<Any?, FirDiagnosticFactory0> {
        return DummyDelegate(FirDiagnosticFactory0(prop.name, severity, positioningStrategy, psiType))
    }
}

class DiagnosticFactory1DelegateProvider<A>(
    private val severity: Severity,
    private val positioningStrategy: SourceElementPositioningStrategy,
    private val psiType: KClass<*>
) {
    operator fun provideDelegate(thisRef: Any?, prop: KProperty<*>): ReadOnlyProperty<Any?, FirDiagnosticFactory1<A>> {
        return DummyDelegate(FirDiagnosticFactory1(prop.name, severity, positioningStrategy, psiType))
    }
}

class DiagnosticFactory2DelegateProvider<A, B>(
    private val severity: Severity,
    private val positioningStrategy: SourceElementPositioningStrategy,
    private val psiType: KClass<*>
) {
    operator fun provideDelegate(thisRef: Any?, prop: KProperty<*>): ReadOnlyProperty<Any?, FirDiagnosticFactory2<A, B>> {
        return DummyDelegate(FirDiagnosticFactory2(prop.name, severity, positioningStrategy, psiType))
    }
}

class DiagnosticFactory3DelegateProvider<A, B, C>(
    private val severity: Severity,
    private val positioningStrategy: SourceElementPositioningStrategy,
    private val psiType: KClass<*>
) {
    operator fun provideDelegate(thisRef: Any?, prop: KProperty<*>): ReadOnlyProperty<Any?, FirDiagnosticFactory3<A, B, C>> {
        return DummyDelegate(FirDiagnosticFactory3(prop.name, severity, positioningStrategy, psiType))
    }
}

class DiagnosticFactory4DelegateProvider<A, B, C, D>(
    private val severity: Severity,
    private val positioningStrategy: SourceElementPositioningStrategy,
    private val psiType: KClass<*>
) {
    operator fun provideDelegate(thisRef: Any?, prop: KProperty<*>): ReadOnlyProperty<Any?, FirDiagnosticFactory4<A, B, C, D>> {
        return DummyDelegate(FirDiagnosticFactory4(prop.name, severity, positioningStrategy, psiType))
    }
}

private const val WARNING = "_WARNING"
private const val ERROR = "_ERROR"

class DeprecationDiagnosticFactory0DelegateProvider(
    private val featureForError: LanguageFeature,
    private val positioningStrategy: SourceElementPositioningStrategy,
    private val psiType: KClass<*>
) {
    operator fun provideDelegate(thisRef: Any?, prop: KProperty<*>): ReadOnlyProperty<Any?, FirDiagnosticFactoryForDeprecation0> {
        val errorFactory = FirDiagnosticFactory0("${prop.name}$ERROR", Severity.ERROR, positioningStrategy, psiType)
        val warningFactory = FirDiagnosticFactory0("${prop.name}$WARNING", Severity.WARNING, positioningStrategy, psiType)
        return DummyDelegate(FirDiagnosticFactoryForDeprecation0(featureForError, warningFactory, errorFactory))
    }
}

class DeprecationDiagnosticFactory1DelegateProvider<A>(
    private val featureForError: LanguageFeature,
    private val positioningStrategy: SourceElementPositioningStrategy,
    private val psiType: KClass<*>
) {
    operator fun provideDelegate(thisRef: Any?, prop: KProperty<*>): ReadOnlyProperty<Any?, FirDiagnosticFactoryForDeprecation1<A>> {
        val errorFactory = FirDiagnosticFactory1<A>("${prop.name}$ERROR", Severity.ERROR, positioningStrategy, psiType)
        val warningFactory = FirDiagnosticFactory1<A>("${prop.name}$WARNING", Severity.WARNING, positioningStrategy, psiType)
        return DummyDelegate(FirDiagnosticFactoryForDeprecation1(featureForError, warningFactory, errorFactory))
    }
}

class DeprecationDiagnosticFactory2DelegateProvider<A, B>(
    private val featureForError: LanguageFeature,
    private val positioningStrategy: SourceElementPositioningStrategy,
    private val psiType: KClass<*>
) {
    operator fun provideDelegate(thisRef: Any?, prop: KProperty<*>): ReadOnlyProperty<Any?, FirDiagnosticFactoryForDeprecation2<A, B>> {
        val errorFactory = FirDiagnosticFactory2<A, B>("${prop.name}$ERROR", Severity.ERROR, positioningStrategy, psiType)
        val warningFactory = FirDiagnosticFactory2<A, B>("${prop.name}$WARNING", Severity.WARNING, positioningStrategy, psiType)
        return DummyDelegate(FirDiagnosticFactoryForDeprecation2(featureForError, warningFactory, errorFactory))
    }
}

class DeprecationDiagnosticFactory3DelegateProvider<A, B, C>(
    private val featureForError: LanguageFeature,
    private val positioningStrategy: SourceElementPositioningStrategy,
    private val psiType: KClass<*>
) {
    operator fun provideDelegate(thisRef: Any?, prop: KProperty<*>): ReadOnlyProperty<Any?, FirDiagnosticFactoryForDeprecation3<A, B, C>> {
        val errorFactory = FirDiagnosticFactory3<A, B, C>("${prop.name}$ERROR", Severity.ERROR, positioningStrategy, psiType)
        val warningFactory = FirDiagnosticFactory3<A, B, C>("${prop.name}$WARNING", Severity.WARNING, positioningStrategy, psiType)
        return DummyDelegate(FirDiagnosticFactoryForDeprecation3(featureForError, warningFactory, errorFactory))
    }
}

class DeprecationDiagnosticFactory4DelegateProvider<A, B, C, D>(
    private val featureForError: LanguageFeature,
    private val positioningStrategy: SourceElementPositioningStrategy,
    private val psiType: KClass<*>
) {
    operator fun provideDelegate(thisRef: Any?, prop: KProperty<*>): ReadOnlyProperty<Any?, FirDiagnosticFactoryForDeprecation4<A, B, C, D>> {
        val errorFactory = FirDiagnosticFactory4<A, B, C, D>("${prop.name}$ERROR", Severity.ERROR, positioningStrategy, psiType)
        val warningFactory = FirDiagnosticFactory4<A, B, C, D>("${prop.name}$WARNING", Severity.WARNING, positioningStrategy, psiType)
        return DummyDelegate(FirDiagnosticFactoryForDeprecation4(featureForError, warningFactory, errorFactory))
    }
}


private class DummyDelegate<T>(val value: T) : ReadOnlyProperty<Any?, T> {
    override fun getValue(thisRef: Any?, property: KProperty<*>): T {
        return value
    }
}
