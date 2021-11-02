data class LinePosition(val posA : Int, val posB : Int) {
    override fun toString() = "($posA; $posB)"
}

object FastLCS {

    /**
     * Optimized version of Array<Array<Short>>
     *
     * Data lies in one continuous array
     *
     * Usage:
     * val a = ShortMatrix(n, m)
     * a[0, 0] = 1
     * println(a[0, 0])
     */
    class ShortMatrix(val n: Int, val m: Int) {
        var arr = ShortArray(n * m)
        operator fun get(i: Int, j: Int) : Short = arr[i * m + j]
        operator fun set(i: Int, j: Int, b: Short) {
            arr[i * m + j] = b
        }
        operator fun set(i: Int, j: Int, b: Int) {
            arr[i * m + j] = b.toShort()
        }
    }

    /**
     * Dynamic programming for finding LCS length for two arrays
     *
     * Works in O(|a| * |b|), where a, b are the given arrays
     * [Algorithm on Wikipedia](https://en.wikipedia.org/wiki/Longest_common_subsequence_problem#Solution_for_two_sequences)
     * Returns the computed dp table
     *
     * Some time measurements are available in Benchmarks.txt
     *
     * Usage:
     * println(lcsDP(arrayOf(4, 1, 5, 6, 2, 7, 3), arrayOf(1, 9, 10, 11, 2, 3, 8))[7, 7])
     */
    fun lcsDP(a: LongArray, b: LongArray) : ShortMatrix {
        val n = a.size
        val m = b.size
        val dp = ShortMatrix(n + 1, m + 1)
        for (row in 1..n) {
            for (column in 1..m) {
                if (a[row - 1] == b[column - 1]) {
                    dp[row, column] = dp[row - 1, column - 1] + 1
                } else if (dp[row - 1, column] > dp[row, column - 1]) {
                    dp[row, column] = dp[row - 1, column]
                } else {
                    dp[row, column] = dp[row, column - 1]
                }
            }
        }
        return dp
    }

    /**
     * LCS length for two arrays
     *
     * Runs lcsDP and takes last value from the table
     *
     * Usage:
     * assertEquals(3, lcs(arrayOf(4, 1, 5, 6, 2, 7, 3), arrayOf(1, 9, 10, 11, 2, 3, 8))
     */
    fun lcs(a: LongArray, b: LongArray) : Int {
        return lcsDP(a, b)[a.size, b.size].toInt()
    }

    /**
     * Optimally matches elements from two arrays
     *
     * Firstly, it runs lcsDP and recovers path to optimal answer
     * Then in creates diff = {(posA, posB) for each line in A U B},
     * where posA = index of the line in A or -1, posB = same for B
     *
     * Usage:
     * assertEquals(arrayOf(LinePosition(1, -1), LinePosition(2, 2), LinePosition(1, 3), LinePosition(2, 2)), diff(arrayOf(1, 3, 2), arrayOf(3, 4))
     */
    fun diff(a: LongArray, b: LongArray) : List<LinePosition> {
        val diff = mutableListOf<LinePosition>()

        fun addFromA(i: Int, j: Int) = diff.add(LinePosition(i, 0))
        fun addFromB(i: Int, j: Int) = diff.add(LinePosition(0, j))
        fun addFromBoth(i: Int, j: Int) = diff.add(LinePosition(i, j))
        val dp = lcsDP(a, b)
        var i = a.size
        var j = b.size
        while (i > 0 && j > 0) {
            if (dp[i, j] == dp[i - 1, j]) {
                addFromA(i, j)
                --i
            } else if (dp[i, j] == dp[i, j - 1]) {
                addFromB(i, j)
                --j
            } else {
                addFromBoth(i, j)
                --i
                --j
            }
        }
        while (i > 0) {
            addFromA(i, j)
            --i
        }
        while (j > 0) {
            addFromB(i, j)
            --j
        }
        return diff.reversed()
    }
}

import kotlin.math.min

/**
 * Calculates polynomial hashes for substrings
 *
 * base = BASE and mod = 2^64 (Long overflow)
 */
class PolynomialHasher(arr: LongArray = longArrayOf(), BASE : Long = 263) {
    val prefixHash : List<Long>
    val basePowers : List<Long>
    init {
        var buff = MutableList<Long>(arr.size + 1) {0}
        for (i in arr.indices) {
            buff[i + 1] = buff[i] * BASE + arr[i]
        }
        prefixHash = buff
        buff = MutableList<Long>(arr.size + 1) {1}
        for (i in 1 until buff.size) {
            buff[i] = buff[i - 1] * BASE
        }
        basePowers = buff
    }

    /**
     * Hash for arr[l..r-1]
     */
    fun rangeHash(l: Int, r: Int) : Long {
        return prefixHash[r] - prefixHash[l] * basePowers[r - l]
    }
}

/**
 * Finds approximate diff in (O((N + M)log(N + M)) * map complexity)
 * Approximate means it isn't always optimal, but it still should be correct
 *
 * Algorithm:
 * Find large common substring, assume it wasn't edited and split recursively to left and right parts from it
 * If one of arrays becomes empty answer is trivial
 *
 * Usage:
 * assertEqual(listOf(LinePosition(0, 1), LinePosition(1, 0), LinePosition(2, 2)), HeuristicLCS.diff(longArrayOf(1, 3), longArrayOf(2, 3)))
 */
object HeuristicLCS {
    var hasherA = PolynomialHasher()
    var hasherB = PolynomialHasher()
    var answer = mutableListOf<LinePosition>()
    var arrA = longArrayOf()
    var arrB = longArrayOf()
    var currentLen = 0

    fun diff(arrA: LongArray, arrB: LongArray) : List<LinePosition> {
        this.arrA = arrA
        this.arrB = arrB
        hasherA = PolynomialHasher(arrA)
        hasherB = PolynomialHasher(arrB)
        answer = mutableListOf<LinePosition>()
        currentLen = min(arrA.size, arrB.size)
        solve(0, arrA.size, 0, arrB.size, min(arrA.size, arrB.size))
        return answer
    }

    /**
     * Recursively find diff for a[la..ra-1] and b[lb..rb-1], where largest common substring is less or equal to maxCommon
     *
     * Return nothing and puts found differences to a list of LinePosition (answer)
     */
    fun solve(la: Int, ra: Int, lb: Int, rb: Int, maxCommon: Int) {
//        println("solve $la $ra $lb $rb")
        if (la == ra) {
//            println("empty a")
            for (i in lb until rb) {
                answer.add(LinePosition(0, i + 1))
            }
        } else if (lb == rb) {
//            println("empty b")
            for (i in la until ra) {
                answer.add(LinePosition(i + 1, 0))
            }
        } else {
            assert(la < ra && lb < rb)
            val start = commonSubstring(la, ra, lb, rb, maxCommon)
            val maxLength = min(ra - start.posA, rb - start.posB)
            var length = 0
            assert(la <= start.posA && lb <= start.posB)
//            println("common: ${start.posA} ${start.posB} $length")
            while (length < maxLength && arrA[start.posA + length] == arrB[start.posB + length]) {
                ++length
            }
            solve(la, start.posA, lb, start.posB, min(length, min(start.posA - la, start.posB - lb)))
            assert(start.posA + length <= ra && start.posB + length <= rb)
            for (i in 1..length) {
                answer.add(LinePosition(start.posA + i, start.posB + i))
            }
            solve(start.posA + length, ra, start.posB + length, rb, min(length, min(ra - start.posA, rb - start.posB) - length))
        }
    }

    /**
     * Find (approximately) the largest common substring for two given substrings
     *
     * Returns the beginning of the largest common substring (length is ignored, because it anyway must be checked without hashes)
     */
    fun commonSubstring(la: Int, ra: Int, lb: Int, rb: Int, maxCommon: Int) : LinePosition {
        while (currentLen > maxCommon) {
            currentLen = currentLen * 2 / 3
        }
        while (true) {
            val start = findByLen(la, ra, lb, rb)
            if (start == null) {
                currentLen = currentLen * 2 / 3
            } else {
                return start
            }
        }
    }

    /**
     * Tries to find common substring with length = currentLen
     * Returns it's beginning or null
     */
    fun findByLen(la: Int, ra: Int, lb: Int, rb: Int) : LinePosition? {
        val hash2PosA = mutableMapOf<Long, Int>()
        for (i in la..ra-currentLen) {
            hash2PosA[hasherA.rangeHash(i, i + currentLen)] = i
        }
        for (i in lb..rb-currentLen) {
            val posA = hash2PosA[hasherB.rangeHash(i, i + currentLen)]
            if (posA != null) {
                return LinePosition(posA, i)
            }
        }
        return null
    }
}

/**
 * Testing i-th bit in a
 *
 * Note that i must be in [0, 32)
 *
 * Usage:
 * assertEquals(true, testBit(11, 1))
 * assertEquals(false, testBit(11, 2))
 */
fun testBit(a : Int, i : Int) : Boolean {
    return ((a shr i) and 1) != 0
}

/**
 * Slow algorithm for finding LCS length for two arrays
 *
 * Works in O(2 ^ min(|a|, |b|) * max(|a|, |b|)), where a, b are the given arrays
 * 1. Select a subsequence of the first array
 * 2. Try to greedily match it with elements of b
 *
 * Usage:
 * assertEquals(3, lcsBaseline(arrayOf(4, 1, 5, 6, 2, 7, 3), arrayOf(1, 9, 10, 11, 2, 3, 8))
 */
fun lcsBaseline(a: LongArray, b: LongArray) : Int {
    var small = a
    var big = b
    if (small.size > big.size) {
        small = big.also { big = small }
    }
    assert(a.size <= 20)
    val n = a.size
    var bestAns = 0
    for (mask in 0 until (1 shl n)) {
        var ptr = 0
        var ans = 0
        for (i in small.indices) {
            if (!testBit(mask, i)) {
                continue
            }
            while (ptr < big.size && big[ptr] != small[i]) {
                ++ptr
            }
            if (ptr < big.size && big[ptr] == small[i]) {
                ++ans
                ++ptr
            } else {
                break
            }
        }
        bestAns = Integer.max(bestAns, ans)
    }
    return bestAns
}

import java.io.File
import kotlin.system.exitProcess

/** TODO
 * ND algorithm
 * Block edit
 */



/** Splits text into parts ending with one of delimiters
 *
 * Not the same as String.split, because it can preserve delimiters
 * * means any symbol
 * And if delimiters are ' ' or '\n', they are added to the last part (for correct diff output)
 *
 * Usage:
 * assertEquals(listOf("one;", " two;", " three"), split("one; two; three", ";"))
 * assertEquals(listOf("one;", " ", "two;", " ", "three"), split("one; two; three", "; "))
 */
fun mySplit(s: String, delimiters: Set<Char>, ignoreDelim: Boolean) : List<String> {
    val parts = mutableListOf<String>()
    val buff = StringBuilder()
    val star = '*' in delimiters
    for (c in s) {
        val isDelim = c in delimiters
        if (!isDelim || !ignoreDelim || star) {
            buff.append(c)
        }
        if (star || c in delimiters) {
            if (buff.length > 0) {
                parts.add(buff.toString())
            }
            buff.clear()
        }
    }
    if (buff.length > 0) {
        if (!ignoreDelim) {
            if ('\n' in delimiters) {
                buff.append('\n')
            } else if (' ' in delimiters) {
                buff.append(' ')
            }
        }
        parts.add(buff.toString())
    }
    return parts
}

/**
 * Polynomial 64-bit string hash
 *
 * Namely, it equals (sum (s_i * BASE ^ i)) mod MOD
 *
 * Usage:
 * assertNotEquals(longHash("abacaba"), longHash("abracadabra"))
 * assertEquals(longHash("abacaba"), longHash("abacaba"))
 */
fun longHash(s: String) : Long {
    val base : Long = 59
    val mod : Long = 1e17.toLong() + 3
    var sum : Long = 0
    for (c in s) {
        sum = (base * sum + c.code.toLong()) % mod
    }
    return sum
}

/**
 * Replace lines of the text with their hash values
 *
 * Usage:
 * assert(toHashArray(arrayOf("a", b")) contentEquals arrayOf(97, 98))
 */
fun toHashArray(lines: List<String>) : LongArray {
    val hashList = mutableListOf<Long>()
    for (s in lines) {
        hashList.add(longHash(s))
    }
    return hashList.toLongArray()
}

/**
 * Just prints help
 */
fun printHelp() {
    println("""
                                                                Usage: diff [OPTION]... FILES
                                                                              (C) diff --help
This program compares files line by line or by any other unit you can define with a regex.
There should be exactly two files and any number of options which begin with a hyphen.
File order matters while option order do not.
-c, --color                     use colors instead of '<', '>' and '='
-i, --input-delim=CHARS         splits input into parts ending with CHARS
-o, --output-delim=STRING       joins output with STRING
-n, --ignore-delim              removes delimiters while splitting
-h, --help                      display this help and exit
-g, --ignore-case               convert all input to lowercase before comparing
-f, --fast                      use fast approximation algorithm
Usage:
diff A B                        plain text line-by-line diff for A and B
diff -n --color A B             colored diff without newlines
diff -i=" " -o="\n" A B         word-by-word diff with one word at line
diff -i="*" -o="" A B           char-by-char diff
diff -n -c -i=" " -o=" " A B    colored word-by-word diff without spaces
diff A --input-delim=".?!" --output-delim=";\t" B --color
                                colored sentence-by-sentence diff
                                for A and B with output separated by ";\t"
    """
    )
}

enum class Color(val code: String) {
    Reset("\u001B[0m"),
    Red("\u001B[31m"),
    Green("\u001B[32m");
    override fun toString() : String {
        return code
    }
}

data class CommandlineArguments(
    val files : Pair<String, String>,
    val colorOutput : Boolean,
    val ignoreCase : Boolean,
    val ignoreDelim : Boolean,
    val inputDelim : Set<Char>,
    val outputDelim : String,
    val fastMode : Boolean
)

fun parseArgs(args: Array<String>) : CommandlineArguments {
    val files = mutableListOf<String>()
    var colorOutput = false
    var ignoreCase = false
    var ignoreDelim = false
    var inputDelim = setOf('\n')
    var outputDelim = ""
    var fastMode = false

    for (arg in args) {
        when (arg.substringBefore("=").trim()) {
            "-c", "--color" -> colorOutput = true
            "-g", "--ignore-case" -> ignoreCase = true
            "-n", "--ignore-delim" -> ignoreDelim = true
            "-i", "--input-delim" -> inputDelim = arg.substringAfter("=").toSet()
            "-o", "--output-delim" -> outputDelim = arg.substringAfter("=").trim('"')
            "-f", "--fast" -> fastMode = true
            "-h", "--help" -> printHelp().also{ exitProcess(0) }
            else -> {
                assert(arg.length > 0 && arg.first() != '-')
                files.add(arg)
            }
        }
    }
    assert(files.size == 2)
    return CommandlineArguments(Pair(files[0], files[1]), colorOutput, ignoreCase, ignoreDelim, inputDelim, outputDelim, fastMode)
}

fun readInputFromFile(name: String, args : CommandlineArguments) : List<String> {
    val raw : String = File(name).readText()
    val text : String = if (args.ignoreCase) raw.map{ it.lowercaseChar() }.toString() else raw
    val tokens : List<String> = mySplit(text, args.inputDelim, args.ignoreDelim)
    return tokens
}

fun printDiff(tokensA : List<String>, tokensB : List<String>, result : List<LinePosition>, args : CommandlineArguments) {
    for ((i, j) in result) {
        if (i > 0 && j > 0) {
            print(if (args.colorOutput) Color.Reset else "=")
            print("${tokensA[i - 1]}${args.outputDelim}")
        } else if (i > 0) {
            print(if (args.colorOutput) Color.Red else "<")
            print("${tokensA[i - 1]}${args.outputDelim}")
        } else if (j > 0) {
            print(if (args.colorOutput) Color.Green else ">")
            print("${tokensB[j - 1]}${args.outputDelim}")
        } else {
            assert(false) {"Every line should come from somewhere"}
        }
        if (args.colorOutput) {
            print(Color.Reset)
        }
    }
}

fun main(rawArgs: Array<String>) {
    val args = parseArgs(rawArgs)
    val tokensA = readInputFromFile(args.files.first, args)
    val tokensB = readInputFromFile(args.files.second, args)
    val arrA = toHashArray(tokensA)
    val arrB = toHashArray(tokensB)

    val result = if (args.fastMode) HeuristicLCS.diff(arrA, arrB) else FastLCS.diff(arrA, arrB)
    printDiff(tokensA, tokensB, result, args)
}

