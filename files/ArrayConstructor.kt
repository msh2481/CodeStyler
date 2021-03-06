import kotlin.*

const val intArray1 = IntArray(42).<!EVALUATED: `42`!>size<!>
const val intArray2 = <!EVALUATED: `0`!>IntArray(42)[0]<!>
const val intArray3 = <!EVALUATED: `42`!>IntArray(10) { 42 }[0]<!>
const val intArray4 = <!EVALUATED: `7`!>IntArray(10) { it -> it }[7]<!>

const val floatArray1 = FloatArray(42).<!EVALUATED: `42`!>size<!>
const val floatArray2 = <!EVALUATED: `0.0`!>FloatArray(42)[0]<!>
const val floatArray3 = <!EVALUATED: `42.5`!>FloatArray(10) { 42.5f }[0]<!>
const val floatArray4 = <!EVALUATED: `7.0`!>FloatArray(10) { it -> it.toFloat() }[7]<!>

const val booleanArray1 = BooleanArray(42).<!EVALUATED: `42`!>size<!>
const val booleanArray2 = <!EVALUATED: `false`!>BooleanArray(42)[0]<!>
const val booleanArray3 = <!EVALUATED: `true`!>BooleanArray(10) { true }[0]<!>
const val booleanArray4 = <!EVALUATED: `true`!>BooleanArray(10) { it -> it != 0 }[7]<!>

const val charArray1 = CharArray(42).<!EVALUATED: `42`!>size<!>
const val charArray2 = <!EVALUATED: ` `!>CharArray(42)[0]<!>
const val charArray3 = <!EVALUATED: `4`!>CharArray(10) { '4' }[0]<!>
const val charArray4 = <!EVALUATED: `0`!>CharArray(50) { it -> it.toChar() }[48]<!>

const val array = Array<Any?>(4) {
    when(it) {
        0 -> 1
        1 -> 2.0
        2 -> "3"
        3 -> null
        else -> throw IllegalArgumentException("$it is wrong")
    }
}.<!EVALUATED: `1 2.0 3 null`!>let { it[0].toString() + " " + it[1] + " " + it[2] + " " + it[3] }<!>

@CompileTimeCalculation // can't be marked as const, but can be used in compile time evaluation
val Int.foo: Int get() = this shl 1

const val arrayWithPropertyAtInit = IntArray(3, Int::foo).<!EVALUATED: `0 2 4`!>let { it[0].toString() + " " + it[1] + " " + it[2] }<!>
