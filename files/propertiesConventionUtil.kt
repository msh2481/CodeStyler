/*
 * Copyright 2010-2020 JetBrains s.r.o. and Kotlin Programming Language contributors.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the license/LICENSE.txt file.
 */

package org.jetbrains.kotlin.load.java

import org.jetbrains.kotlin.load.java.BuiltinSpecialProperties.getPropertyNameCandidatesBySpecialGetterName
import org.jetbrains.kotlin.name.Name
import org.jetbrains.kotlin.util.capitalizeDecapitalize.decapitalizeSmartForCompiler

fun propertyNameByGetMethodName(methodName: Name): Name? =
    propertyNameFromAccessorMethodName(methodName, "get") ?: propertyNameFromAccessorMethodName(methodName, "is", removePrefix = false)

fun propertyNameBySetMethodName(methodName: Name, withIsPrefix: Boolean): Name? =
    propertyNameFromAccessorMethodName(methodName, "set", addPrefix = if (withIsPrefix) "is" else null)

fun propertyNamesBySetMethodName(methodName: Name): List<Name> =
    listOfNotNull(propertyNameBySetMethodName(methodName, false), propertyNameBySetMethodName(methodName, true))

fun propertyNamesByAccessorName(name: Name): List<Name> = listOfNotNull(
    propertyNameByGetMethodName(name),
    propertyNameBySetMethodName(name, withIsPrefix = true),
    propertyNameBySetMethodName(name, withIsPrefix = false)
)

private fun propertyNameFromAccessorMethodName(
    methodName: Name,
    prefix: String,
    removePrefix: Boolean = true,
    addPrefix: String? = null
): Name? {
    if (methodName.isSpecial) return null
    val identifier = methodName.identifier
    if (!identifier.startsWith(prefix)) return null
    if (identifier.length == prefix.length) return null
    if (identifier[prefix.length] in 'a'..'z') return null

    if (addPrefix != null) {
        assert(removePrefix)
        return Name.identifier(addPrefix + identifier.removePrefix(prefix))
    }

    if (!removePrefix) return methodName
    val name = identifier.removePrefix(prefix).decapitalizeSmartForCompiler(asciiOnly = true)
    if (!Name.isValidIdentifier(name)) return null
    return Name.identifier(name)
}

fun getPropertyNamesCandidatesByAccessorName(name: Name): List<Name> {
    val nameAsString = name.asString()

    if (JvmAbi.isGetterName(nameAsString)) {
        return listOfNotNull(propertyNameByGetMethodName(name))
    }

    if (JvmAbi.isSetterName(nameAsString)) {
        return propertyNamesBySetMethodName(name)
    }

    return getPropertyNameCandidatesBySpecialGetterName(name)
}
