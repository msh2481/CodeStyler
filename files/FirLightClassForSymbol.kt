/*
 * Copyright 2010-2020 JetBrains s.r.o. and Kotlin Programming Language contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package org.jetbrains.kotlin.light.classes.symbol

import com.intellij.psi.*
import org.jetbrains.kotlin.analysis.api.isValid
import org.jetbrains.kotlin.analysis.api.symbols.*
import org.jetbrains.kotlin.analysis.api.symbols.markers.KtSymbolKind
import org.jetbrains.kotlin.analysis.api.symbols.markers.KtSymbolWithVisibility
import org.jetbrains.kotlin.asJava.builder.LightMemberOriginForDeclaration
import org.jetbrains.kotlin.asJava.classes.METHOD_INDEX_BASE
import org.jetbrains.kotlin.asJava.classes.lazyPub
import org.jetbrains.kotlin.asJava.elements.KtLightField
import org.jetbrains.kotlin.asJava.elements.KtLightMethod
import org.jetbrains.kotlin.builtins.StandardNames.HASHCODE_NAME
import org.jetbrains.kotlin.descriptors.Visibility
import org.jetbrains.kotlin.light.classes.symbol.classes.*
import org.jetbrains.kotlin.load.java.JvmAbi
import org.jetbrains.kotlin.name.Name
import org.jetbrains.kotlin.resolve.DataClassResolver
import org.jetbrains.kotlin.resolve.jvm.diagnostics.JvmDeclarationOriginKind
import org.jetbrains.kotlin.util.OperatorNameConventions.EQUALS
import org.jetbrains.kotlin.util.OperatorNameConventions.TO_STRING
import org.jetbrains.kotlin.utils.addToStdlib.applyIf

internal class FirLightClassForSymbol(
    private val classOrObjectSymbol: KtNamedClassOrObjectSymbol,
    manager: PsiManager
) : FirLightClassForClassOrObjectSymbol(classOrObjectSymbol, manager) {

    init {
        require(classOrObjectSymbol.classKind != KtClassKind.INTERFACE && classOrObjectSymbol.classKind != KtClassKind.ANNOTATION_CLASS)
    }

    internal fun tryGetEffectiveVisibility(symbol: KtCallableSymbol): Visibility? {

        if (symbol !is KtPropertySymbol && symbol !is KtFunctionSymbol) return null

        var visibility = (symbol as? KtSymbolWithVisibility)?.visibility

        analyzeWithSymbolAsContext(symbol) {
            for (overriddenSymbol in symbol.getAllOverriddenSymbols()) {
                val newVisibility = (overriddenSymbol as? KtSymbolWithVisibility)?.visibility
                if (newVisibility != null) {
                    visibility = newVisibility
                }
            }
        }

        return visibility
    }

    private val isTopLevel: Boolean = classOrObjectSymbol.symbolKind == KtSymbolKind.TOP_LEVEL

    private val _modifierList: PsiModifierList? by lazyPub {

        val modifiers = mutableSetOf(classOrObjectSymbol.toPsiVisibilityForClass(isTopLevel))
        classOrObjectSymbol.computeSimpleModality()?.run {
            modifiers.add(this)
        }
        if (!isTopLevel && !classOrObjectSymbol.isInner) {
            modifiers.add(PsiModifier.STATIC)
        }

        val annotations = classOrObjectSymbol.computeAnnotations(
            parent = this@FirLightClassForSymbol,
            nullability = NullabilityType.Unknown,
            annotationUseSiteTarget = null,
        )

        FirLightClassModifierList(this@FirLightClassForSymbol, modifiers, annotations)
    }

    override fun getModifierList(): PsiModifierList? = _modifierList
    override fun getOwnFields(): List<KtLightField> = _ownFields
    override fun getOwnMethods(): List<PsiMethod> = _ownMethods
    override fun getExtendsList(): PsiReferenceList? = _extendsList
    override fun getImplementsList(): PsiReferenceList? = _implementsList

    private val _ownInnerClasses: List<FirLightClassBase> by lazyPub {
        classOrObjectSymbol.createInnerClasses(manager)
    }

    override fun getOwnInnerClasses(): List<PsiClass> = _ownInnerClasses

    private val _extendsList by lazyPub { createInheritanceList(forExtendsList = true, classOrObjectSymbol.superTypes) }
    private val _implementsList by lazyPub { createInheritanceList(forExtendsList = false, classOrObjectSymbol.superTypes) }

    private val _ownMethods: List<KtLightMethod> by lazyPub {

        val result = mutableListOf<KtLightMethod>()

        analyzeWithSymbolAsContext(classOrObjectSymbol) {
            val declaredMemberScope = classOrObjectSymbol.getDeclaredMemberScope()

            val visibleDeclarations = declaredMemberScope.getCallableSymbols().applyIf(isEnum) {
                filterNot { function ->
                    function is KtFunctionSymbol && function.name.asString().let { it == "values" || it == "valueOf" }
                }
            }.applyIf(classOrObjectSymbol.classKind == KtClassKind.OBJECT) {
                filterNot {
                    it is KtKotlinPropertySymbol && it.isConst
                }
            }.applyIf(classOrObjectSymbol.isData) {
                // Technically, synthetic members of `data` class, such as `componentN` or `copy`, are visible.
                // They're just needed to be added later (to be in a backward-compatible order of members).
                filterNot { function ->
                    function is KtFunctionSymbol && function.origin == KtSymbolOrigin.SOURCE_MEMBER_GENERATED
                }
            }

            val suppressStatic = classOrObjectSymbol.classKind == KtClassKind.COMPANION_OBJECT
            createMethods(visibleDeclarations, result, suppressStaticForMethods = suppressStatic)

            createConstructors(declaredMemberScope.getConstructors(), result)
        }

        addMethodsFromCompanionIfNeeded(result)

        addMethodsFromDataClass(result)

        result
    }

    private fun addMethodsFromCompanionIfNeeded(result: MutableList<KtLightMethod>) {
        classOrObjectSymbol.companionObject?.run {
            analyzeWithSymbolAsContext(this) {
                val methods = getDeclaredMemberScope().getCallableSymbols()
                    .filterIsInstance<KtFunctionSymbol>()
                    .filter { it.hasJvmStaticAnnotation() }
                createMethods(methods, result)
            }
        }
    }

    private fun addMethodsFromDataClass(result: MutableList<KtLightMethod>) {
        if (!classOrObjectSymbol.isData) return

        fun createMethodFromAny(ktFunctionSymbol: KtFunctionSymbol) {
            // Similar to `copy`, synthetic members from `Any` should refer to `data` class as origin, not the function in `Any`.
            val lightMemberOrigin = LightMemberOriginForDeclaration(this.kotlinOrigin!!, JvmDeclarationOriginKind.OTHER)
            result.add(
                FirLightSimpleMethodForSymbol(
                    functionSymbol = ktFunctionSymbol,
                    lightMemberOrigin = lightMemberOrigin,
                    containingClass = this,
                    isTopLevel = false,
                    methodIndex = METHOD_INDEX_BASE,
                    suppressStatic = false
                )
            )
        }

        analyzeWithSymbolAsContext(classOrObjectSymbol) {
            val componentAndCopyFunctions = mutableListOf<KtFunctionSymbol>()
            val functionsFromAny = mutableMapOf<Name, KtFunctionSymbol>()
            // NB: componentN and copy are added during RAW FIR, but synthetic members from `Any` are not.
            // Thus, using declared member scope is not sufficient to lookup "all" synthetic members.
            classOrObjectSymbol.getMemberScope().getCallableSymbols().forEach { functionSymbol ->
                if (functionSymbol is KtFunctionSymbol) {
                    val name = functionSymbol.name
                    if (functionSymbol.origin == KtSymbolOrigin.SOURCE_MEMBER_GENERATED &&
                        (DataClassResolver.isCopy(name) || DataClassResolver.isComponentLike(name))
                    ) {
                        componentAndCopyFunctions.add(functionSymbol)
                    }
                    if (functionSymbol.dispatchType?.isAny == true && name.isFromAny) {
                        functionsFromAny[name] = functionSymbol
                    }
                }
            }
            createMethods(componentAndCopyFunctions.asSequence(), result)
            // NB: functions from `Any` are not in an alphabetic order.
            functionsFromAny[TO_STRING]?.let { createMethodFromAny(it) }
            functionsFromAny[HASHCODE_NAME]?.let { createMethodFromAny(it) }
            functionsFromAny[EQUALS]?.let { createMethodFromAny(it) }
        }
    }

    private val Name.isFromAny: Boolean
        get() = this == EQUALS || this == HASHCODE_NAME || this == TO_STRING

    private val _ownFields: List<KtLightField> by lazyPub {

        val result = mutableListOf<KtLightField>()

        addCompanionObjectFieldIfNeeded(result)
        addInstanceFieldIfNeeded(result)

        addFieldsFromCompanionIfNeeded(result)
        addPropertyBackingFields(result)

        result
    }

    private fun addInstanceFieldIfNeeded(result: MutableList<KtLightField>) {
        val isNamedObject = classOrObjectSymbol.classKind == KtClassKind.OBJECT
        if (isNamedObject && classOrObjectSymbol.symbolKind != KtSymbolKind.LOCAL) {
            result.add(
                FirLightFieldForObjectSymbol(
                    objectSymbol = classOrObjectSymbol,
                    containingClass = this@FirLightClassForSymbol,
                    name = JvmAbi.INSTANCE_FIELD,
                    lightMemberOrigin = null
                )
            )
        }
    }

    private fun addFieldsFromCompanionIfNeeded(result: MutableList<KtLightField>) {
        classOrObjectSymbol.companionObject?.run {
            analyzeWithSymbolAsContext(this) {
                getDeclaredMemberScope().getCallableSymbols()
                    .filterIsInstance<KtPropertySymbol>()
                    .filter { it.hasJvmFieldAnnotation() || it.hasJvmStaticAnnotation() || it is KtKotlinPropertySymbol && it.isConst }
                    .mapTo(result) {
                        FirLightFieldForPropertySymbol(
                            propertySymbol = it,
                            fieldName = it.name.asString(),
                            containingClass = this@FirLightClassForSymbol,
                            lightMemberOrigin = null,
                            isTopLevel = false,
                            forceStatic = !it.hasJvmStaticAnnotation(),
                            takePropertyVisibility = true
                        )
                    }
            }
        }
    }

    private fun addPropertyBackingFields(result: MutableList<KtLightField>) {
        analyzeWithSymbolAsContext(classOrObjectSymbol) {
            val propertySymbols = classOrObjectSymbol.getDeclaredMemberScope().getCallableSymbols()
                .filterIsInstance<KtPropertySymbol>()
                .applyIf(classOrObjectSymbol.classKind == KtClassKind.COMPANION_OBJECT) {
                    filterNot { it.hasJvmFieldAnnotation() || it is KtKotlinPropertySymbol && it.isConst }
                }

            val nameGenerator = FirLightField.FieldNameGenerator()
            val isObject = classOrObjectSymbol.classKind == KtClassKind.OBJECT
            val isCompanionObject = classOrObjectSymbol.classKind == KtClassKind.COMPANION_OBJECT

            for (propertySymbol in propertySymbols) {
                val isJvmField = propertySymbol.hasJvmFieldAnnotation()
                val isJvmStatic = propertySymbol.hasJvmStaticAnnotation()
                val isLateInit = (propertySymbol as? KtKotlinPropertySymbol)?.isLateInit == true

                val forceStatic =
                    isObject && (propertySymbol is KtKotlinPropertySymbol && propertySymbol.isConst || isJvmStatic || isJvmField)
                val takePropertyVisibility = !isCompanionObject && (isLateInit || isJvmField || forceStatic)

                createField(
                    declaration = propertySymbol,
                    nameGenerator = nameGenerator,
                    isTopLevel = false,
                    forceStatic = forceStatic,
                    takePropertyVisibility = takePropertyVisibility,
                    result = result
                )
            }

            if (isEnum) {
                classOrObjectSymbol.getDeclaredMemberScope().getCallableSymbols()
                    .filterIsInstance<KtEnumEntrySymbol>()
                    .mapTo(result) { FirLightFieldForEnumEntry(it, this@FirLightClassForSymbol, null) }
            }
        }
    }

    override fun hashCode(): Int = classOrObjectSymbol.hashCode()

    override fun equals(other: Any?): Boolean =
        this === other || (other is FirLightClassForSymbol && classOrObjectSymbol == other.classOrObjectSymbol)

    override fun isInterface(): Boolean = false

    override fun isAnnotationType(): Boolean = false

    override fun isEnum(): Boolean =
        classOrObjectSymbol.classKind == KtClassKind.ENUM_CLASS

    override fun copy(): FirLightClassForSymbol =
        FirLightClassForSymbol(classOrObjectSymbol, manager)

    override fun isValid(): Boolean = super.isValid() && classOrObjectSymbol.isValid()
}
