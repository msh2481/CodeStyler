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

import org.jetbrains.kotlin.ir.symbols.IrValueSymbol

abstract class IrValueAccessExpression : IrDeclarationReference() {
    abstract override val symbol: IrValueSymbol
    abstract val origin: IrStatementOrigin?
}

abstract class IrGetValue : IrValueAccessExpression() {
    abstract fun copyWithOffsets(newStartOffset: Int, newEndOffset: Int): IrGetValue
}

abstract class IrSetValue : IrValueAccessExpression() {
    abstract override val symbol: IrValueSymbol
    abstract var value: IrExpression
}
