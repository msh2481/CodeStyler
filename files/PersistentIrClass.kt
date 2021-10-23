/*
 * Copyright 2010-2021 JetBrains s.r.o. and Kotlin Programming Language contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package org.jetbrains.kotlin.ir.declarations.persistent

import java.util.ArrayList
import java.util.Collections
import org.jetbrains.kotlin.descriptors.ClassDescriptor
import org.jetbrains.kotlin.descriptors.ClassKind
import org.jetbrains.kotlin.descriptors.DescriptorVisibility
import org.jetbrains.kotlin.descriptors.InlineClassRepresentation
import org.jetbrains.kotlin.descriptors.Modality
import org.jetbrains.kotlin.descriptors.SourceElement
import org.jetbrains.kotlin.ir.ObsoleteDescriptorBasedAPI
import org.jetbrains.kotlin.ir.declarations.IrAttributeContainer
import org.jetbrains.kotlin.ir.declarations.IrClass
import org.jetbrains.kotlin.ir.declarations.IrDeclaration
import org.jetbrains.kotlin.ir.declarations.IrDeclarationOrigin
import org.jetbrains.kotlin.ir.declarations.IrDeclarationParent
import org.jetbrains.kotlin.ir.declarations.IrTypeParameter
import org.jetbrains.kotlin.ir.declarations.IrValueParameter
import org.jetbrains.kotlin.ir.declarations.MetadataSource
import org.jetbrains.kotlin.ir.declarations.persistent.carriers.Carrier
import org.jetbrains.kotlin.ir.declarations.persistent.carriers.ClassCarrier
import org.jetbrains.kotlin.ir.expressions.IrConstructorCall
import org.jetbrains.kotlin.ir.symbols.IrClassSymbol
import org.jetbrains.kotlin.ir.symbols.IrTypeParameterSymbol
import org.jetbrains.kotlin.ir.symbols.IrValueParameterSymbol
import org.jetbrains.kotlin.ir.types.IrSimpleType
import org.jetbrains.kotlin.ir.types.IrType
import org.jetbrains.kotlin.ir.util.IdSignature
import org.jetbrains.kotlin.name.Name

// Auto-generated by compiler/ir/ir.tree.persistent/generator/src/org/jetbrains/kotlin/ir/persistentIrGenerator/Main.kt. DO NOT EDIT!

internal class PersistentIrClass(
    override val startOffset: Int,
    override val endOffset: Int,
    origin: IrDeclarationOrigin,
    override val symbol: IrClassSymbol,
    override val name: Name,
    override val kind: ClassKind,
    visibility: DescriptorVisibility,
    modality: Modality,
    override val isCompanion: Boolean = false,
    override val isInner: Boolean = false,
    override val isData: Boolean = false,
    override val isExternal: Boolean = false,
    override val isInline: Boolean = false,
    override val isExpect: Boolean = false,
    override val isFun: Boolean = false,
    override val source: SourceElement = SourceElement.NO_SOURCE,
    override val factory: PersistentIrFactory
) : IrClass(),
    PersistentIrDeclarationBase<ClassCarrier>,
    ClassCarrier {

    init {
        symbol.bind(this)
    }

    override var signature: IdSignature? = factory.currentSignature(this)

    override var lastModified: Int = factory.stageController.currentStage
    override var loweredUpTo: Int = factory.stageController.currentStage
    override var values: Array<Carrier>? = null
    override val createdOn: Int = factory.stageController.currentStage

    override var parentField: IrDeclarationParent? = null
    override var originField: IrDeclarationOrigin = origin
    override var removedOn: Int = Int.MAX_VALUE
    override var annotationsField: List<IrConstructorCall> = emptyList()
    private val hashCodeValue: Int = PersistentIrDeclarationBase.hashCodeCounter++
    override fun hashCode(): Int = hashCodeValue
    override fun equals(other: Any?): Boolean = (this === other)

    @ObsoleteDescriptorBasedAPI
    override val descriptor: ClassDescriptor
        get() = symbol.descriptor

    override var visibilityField: DescriptorVisibility = visibility

    override var visibility: DescriptorVisibility
        get() = getCarrier().visibilityField
        set(v) {
            if (visibility !== v) {
                setCarrier()
                visibilityField = v
            }
        }

    override var thisReceiverField: IrValueParameter? = null

    override var thisReceiverSymbolField: IrValueParameterSymbol?
        get() = thisReceiverField?.symbol
        set(v) {
            thisReceiverField = v?.owner
        }

    override var thisReceiver: IrValueParameter?
        get() = getCarrier().thisReceiverField
        set(v) {
            if (thisReceiver !== v) {
                setCarrier()
                thisReceiverField = v
            }
        }

    private var initialDeclarations: MutableList<IrDeclaration>? = null

    override val declarations: MutableList<IrDeclaration> = ArrayList()
        get() {
            if (createdOn < factory.stageController.currentStage && initialDeclarations == null) {
                initialDeclarations = Collections.unmodifiableList(ArrayList(field))
            }

            return if (factory.stageController.canAccessDeclarationsOf(this)) {
                ensureLowered()
                field
            } else {
                initialDeclarations ?: field
            }
        }

    override var typeParametersField: List<IrTypeParameter> = emptyList()

    override var typeParametersSymbolField: List<IrTypeParameterSymbol>
        get() = typeParametersField.map { it.symbol }
        set(v) {
            typeParametersField = v.map { it.owner }
        }

    override var typeParameters: List<IrTypeParameter>
        get() = getCarrier().typeParametersField
        set(v) {
            if (typeParameters !== v) {
                setCarrier()
                typeParametersField = v
            }
        }

    override var superTypesField: List<IrType> = emptyList()

    override var superTypes: List<IrType>
        get() = getCarrier().superTypesField
        set(v) {
            if (superTypes !== v) {
                setCarrier()
                superTypesField = v
            }
        }

    override var metadata: MetadataSource? = null

    override var modalityField: Modality = modality

    override var modality: Modality
        get() = getCarrier().modalityField
        set(v) {
            if (modality !== v) {
                setCarrier()
                modalityField = v
            }
        }

    override var inlineClassRepresentationField: InlineClassRepresentation<IrSimpleType>? = null

    override var inlineClassRepresentation: InlineClassRepresentation<IrSimpleType>?
        get() = getCarrier().inlineClassRepresentationField
        set(v) {
            if (inlineClassRepresentation !== v) {
                setCarrier()
                inlineClassRepresentationField = v
            }
        }

    override var attributeOwnerId: IrAttributeContainer = this

    override var sealedSubclassesField: List<IrClassSymbol> = emptyList()

    override var sealedSubclasses: List<IrClassSymbol>
        get() = getCarrier().sealedSubclassesField
        set(v) {
            if (sealedSubclasses !== v) {
                setCarrier()
                sealedSubclassesField = v
            }
        }
}
