// Auto-generated by org.jetbrains.kotlin.generators.tests.GenerateRangesCodegenTestData. DO NOT EDIT!
// WITH_RUNTIME


val MaxUI = UInt.MAX_VALUE
val MaxUB = UByte.MAX_VALUE
val MaxUS = UShort.MAX_VALUE
val MaxUL = ULong.MAX_VALUE

fun box(): String {
    val list1 = ArrayList<UInt>()
    for (i in (MaxUI - 5u)..MaxUI step 3) {
        list1.add(i)
        if (list1.size > 23) break
    }
    if (list1 != listOf<UInt>(MaxUI - 5u, MaxUI - 2u)) {
        return "Wrong elements for (MaxUI - 5u)..MaxUI step 3: $list1"
    }

    val list2 = ArrayList<UInt>()
    for (i in (MaxUB - 5u).toUByte()..MaxUB step 3) {
        list2.add(i)
        if (list2.size > 23) break
    }
    if (list2 != listOf<UInt>((MaxUB - 5u).toUInt(), (MaxUB - 2u).toUInt())) {
        return "Wrong elements for (MaxUB - 5u).toUByte()..MaxUB step 3: $list2"
    }

    val list3 = ArrayList<UInt>()
    for (i in (MaxUS - 5u).toUShort()..MaxUS step 3) {
        list3.add(i)
        if (list3.size > 23) break
    }
    if (list3 != listOf<UInt>((MaxUS - 5u).toUInt(), (MaxUS - 2u).toUInt())) {
        return "Wrong elements for (MaxUS - 5u).toUShort()..MaxUS step 3: $list3"
    }

    val list4 = ArrayList<ULong>()
    for (i in (MaxUL - 5u)..MaxUL step 3) {
        list4.add(i)
        if (list4.size > 23) break
    }
    if (list4 != listOf<ULong>((MaxUL - 5u), (MaxUL - 2u))) {
        return "Wrong elements for (MaxUL - 5u)..MaxUL step 3: $list4"
    }

    return "OK"
}
