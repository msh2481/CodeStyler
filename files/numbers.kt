/*
 * Copyright 2010-2020 JetBrains s.r.o. and Kotlin Programming Language contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package kotlin

import kotlin.wasm.internal.*

/**
 * Counts the number of set bits in the binary representation of this [Int] number.
 */
@WasmOp(WasmOp.I32_POPCNT)
public actual fun Int.countOneBits(): Int =
    implementedAsIntrinsic

/**
 * Counts the number of consecutive most significant bits that are zero in the binary representation of this [Int] number.
 */
@WasmOp(WasmOp.I32_CLZ)
public actual fun Int.countLeadingZeroBits(): Int =
    implementedAsIntrinsic

/**
 * Counts the number of consecutive least significant bits that are zero in the binary representation of this [Int] number.
 */
@WasmOp(WasmOp.I32_CTZ)
public actual fun Int.countTrailingZeroBits(): Int =
    implementedAsIntrinsic

/**
 * Returns a number having a single bit set in the position of the most significant set bit of this [Int] number,
 * or zero, if this number is zero.
 */
public actual fun Int.takeHighestOneBit(): Int =
    if (this == 0) 0 else 1.shl(32 - 1 - countLeadingZeroBits())

/**
 * Returns a number having a single bit set in the position of the least significant set bit of this [Int] number,
 * or zero, if this number is zero.
 */
public actual fun Int.takeLowestOneBit(): Int =
    this and -this

/**
 * Rotates the binary representation of this [Int] number left by the specified [bitCount] number of bits.
 * The most significant bits pushed out from the left side reenter the number as the least significant bits on the right side.
 *
 * Rotating the number left by a negative bit count is the same as rotating it right by the negated bit count:
 * `number.rotateLeft(-n) == number.rotateRight(n)`
 *
 * Rotating by a multiple of [Int.SIZE_BITS] (32) returns the same number, or more generally
 * `number.rotateLeft(n) == number.rotateLeft(n % 32)`
 */
@ExperimentalStdlibApi
public actual fun Int.rotateLeft(bitCount: Int): Int =
    shl(bitCount) or ushr(32 - bitCount)


/**
 * Rotates the binary representation of this [Int] number right by the specified [bitCount] number of bits.
 * The least significant bits pushed out from the right side reenter the number as the most significant bits on the left side.
 *
 * Rotating the number right by a negative bit count is the same as rotating it left by the negated bit count:
 * `number.rotateRight(-n) == number.rotateLeft(n)`
 *
 * Rotating by a multiple of [Int.SIZE_BITS] (32) returns the same number, or more generally
 * `number.rotateRight(n) == number.rotateRight(n % 32)`
 */
@ExperimentalStdlibApi
public actual fun Int.rotateRight(bitCount: Int): Int =
    shl(32 - bitCount) or ushr(bitCount)


/**
 * Counts the number of set bits in the binary representation of this [Long] number.
 */
public actual inline fun Long.countOneBits(): Int =
    wasm_i64_popcnt(this).toInt()

/**
 * Counts the number of consecutive most significant bits that are zero in the binary representation of this [Long] number.
 */
@ExperimentalStdlibApi
public actual fun Long.countLeadingZeroBits(): Int = wasm_i64_clz(this).toInt()

/**
 * Counts the number of consecutive least significant bits that are zero in the binary representation of this [Long] number.
 */
public actual inline fun Long.countTrailingZeroBits(): Int =
    wasm_i64_ctz(this).toInt()

/**
 * Returns a number having a single bit set in the position of the most significant set bit of this [Long] number,
 * or zero, if this number is zero.
 */
public actual fun Long.takeHighestOneBit(): Long =
    if (this == 0L) 0L else 1L.shl(64 - 1 - countLeadingZeroBits())

/**
 * Returns a number having a single bit set in the position of the least significant set bit of this [Long] number,
 * or zero, if this number is zero.
 */
public actual fun Long.takeLowestOneBit(): Long =
    this and -this

/**
 * Rotates the binary representation of this [Long] number left by the specified [bitCount] number of bits.
 * The most significant bits pushed out from the left side reenter the number as the least significant bits on the right side.
 *
 * Rotating the number left by a negative bit count is the same as rotating it right by the negated bit count:
 * `number.rotateLeft(-n) == number.rotateRight(n)`
 *
 * Rotating by a multiple of [Long.SIZE_BITS] (64) returns the same number, or more generally
 * `number.rotateLeft(n) == number.rotateLeft(n % 64)`
 */
@ExperimentalStdlibApi
public actual fun Long.rotateLeft(bitCount: Int): Long =
    shl(bitCount) or ushr(64 - bitCount)

/**
 * Rotates the binary representation of this [Long] number right by the specified [bitCount] number of bits.
 * The least significant bits pushed out from the right side reenter the number as the most significant bits on the left side.
 *
 * Rotating the number right by a negative bit count is the same as rotating it left by the negated bit count:
 * `number.rotateRight(-n) == number.rotateLeft(n)`
 *
 * Rotating by a multiple of [Long.SIZE_BITS] (64) returns the same number, or more generally
 * `number.rotateRight(n) == number.rotateRight(n % 64)`
 */
@ExperimentalStdlibApi
@kotlin.internal.InlineOnly
public actual inline fun Long.rotateRight(bitCount: Int): Long =
    shl(64 - bitCount) or ushr(bitCount)