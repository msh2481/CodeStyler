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

package org.jetbrains.kotlin.resolve.calls.components

import org.jetbrains.kotlin.builtins.KotlinBuiltIns
import org.jetbrains.kotlin.descriptors.ModuleDescriptor
import org.jetbrains.kotlin.resolve.calls.inference.components.ConstraintInjector
import org.jetbrains.kotlin.resolve.calls.model.KotlinCallArgument
import org.jetbrains.kotlin.resolve.calls.components.candidate.ResolutionCandidate
import org.jetbrains.kotlin.resolve.calls.model.ResolvedCallArgument
import org.jetbrains.kotlin.resolve.calls.results.*
import org.jetbrains.kotlin.types.KotlinType
import org.jetbrains.kotlin.types.checker.KotlinTypeRefiner
import org.jetbrains.kotlin.util.CancellationChecker
import java.util.*

class NewOverloadingConflictResolver(
    builtIns: KotlinBuiltIns,
    module: ModuleDescriptor,
    specificityComparator: TypeSpecificityComparator,
    platformOverloadsSpecificityComparator: PlatformOverloadsSpecificityComparator,
    cancellationChecker: CancellationChecker,
    statelessCallbacks: KotlinResolutionStatelessCallbacks,
    constraintInjector: ConstraintInjector,
    kotlinTypeRefiner: KotlinTypeRefiner,
) : OverloadingConflictResolver<ResolutionCandidate>(
    builtIns,
    module,
    specificityComparator,
    platformOverloadsSpecificityComparator,
    cancellationChecker,
    {
        // todo investigate
        it.resolvedCall.candidateDescriptor
    },
    { statelessCallbacks.createConstraintSystemForOverloadResolution(constraintInjector, builtIns) },
    Companion::createFlatSignature,
    { it.variableCandidateIfInvoke },
    { statelessCallbacks.isDescriptorFromSource(it) },
    { it.resolvedCall.hasSamConversion },
    kotlinTypeRefiner,
) {

    companion object {
        private fun createFlatSignature(candidate: ResolutionCandidate): FlatSignature<ResolutionCandidate> {

            val resolvedCall = candidate.resolvedCall
            val originalDescriptor = resolvedCall.candidateDescriptor.original
            val originalValueParameters = originalDescriptor.valueParameters

            var numDefaults = 0
            val valueArgumentToParameterType = HashMap<KotlinCallArgument, KotlinType>()
            for ((valueParameter, resolvedValueArgument) in resolvedCall.argumentMappingByOriginal) {
                if (resolvedValueArgument is ResolvedCallArgument.DefaultArgument) {
                    numDefaults++
                } else {
                    val originalValueParameter = originalValueParameters[valueParameter.index]
                    for (valueArgument in resolvedValueArgument.arguments) {
                        valueArgumentToParameterType[valueArgument] = candidate.resolvedCall.argumentsWithConversion[valueArgument]?.convertedTypeByOriginParameter ?:
                                valueArgument.getExpectedType(originalValueParameter, candidate.callComponents.languageVersionSettings)
                    }
                }
            }

            return FlatSignature.create(candidate,
                                        originalDescriptor,
                                        numDefaults,
                                        resolvedCall.atom.argumentsInParenthesis.map { valueArgumentToParameterType[it] } +
                                                listOfNotNull(resolvedCall.atom.externalArgument?.let { valueArgumentToParameterType[it] })
            )

        }
    }
}



