/*
 * Copyright 2010-2016 JetBrains s.r.o.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.jetbrains.kotlin.ir.expressions

import org.jetbrains.kotlin.ir.IrStatement
import org.jetbrains.kotlin.ir.declarations.IrReturnTarget
import org.jetbrains.kotlin.ir.declarations.IrSymbolOwner
import org.jetbrains.kotlin.ir.symbols.IrFileSymbol
import org.jetbrains.kotlin.ir.symbols.IrFunctionSymbol
import org.jetbrains.kotlin.ir.symbols.IrReturnableBlockSymbol
import org.jetbrains.kotlin.ir.util.fileOrNull
import org.jetbrains.kotlin.ir.util.transformInPlace
import org.jetbrains.kotlin.ir.visitors.IrElementTransformer
import org.jetbrains.kotlin.ir.visitors.IrElementVisitor

abstract class IrContainerExpression : IrExpression(), IrStatementContainer {
    abstract val origin: IrStatementOrigin?
    abstract val isTransparentScope: Boolean

    override val statements: MutableList<IrStatement> = ArrayList(2)

    override fun <D> acceptChildren(visitor: IrElementVisitor<Unit, D>, data: D) {
        statements.forEach { it.accept(visitor, data) }
    }

    override fun <D> transformChildren(transformer: IrElementTransformer<D>, data: D) {
        statements.transformInPlace(transformer, data)
    }
}

abstract class IrBlock : IrContainerExpression() {
    override val isTransparentScope: Boolean
        get() = false
}

abstract class IrComposite : IrContainerExpression() {
    override val isTransparentScope: Boolean
        get() = true
}

abstract class IrReturnableBlock : IrBlock(), IrSymbolOwner, IrReturnTarget {
    abstract override val symbol: IrReturnableBlockSymbol

    abstract val inlineFunctionSymbol: IrFunctionSymbol?
}

val IrReturnableBlock.sourceFileSymbol: IrFileSymbol?
    get() = inlineFunctionSymbol?.owner?.fileOrNull?.symbol
