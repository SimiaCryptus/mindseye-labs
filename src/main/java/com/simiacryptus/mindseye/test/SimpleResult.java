/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.test;

import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.lang.ReferenceCounting;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.UUID;

public interface SimpleResult extends ReferenceCounting {
  @Nonnull
  TensorList[] getInputDerivative();

  @Nonnull
  DeltaSet<UUID> getLayerDerivative();

  @Nonnull
  TensorList getOutput();

  @Nullable
  static @SuppressWarnings("unused")
  SimpleResult[] addRef(@Nullable SimpleResult[] array) {
    return RefUtil.addRef(array);
  }

  @Nullable
  static @SuppressWarnings("unused")
  SimpleResult[][] addRef(@Nullable SimpleResult[][] array) {
    return RefUtil.addRef(array);
  }

  void _free();

  @Nonnull
  SimpleResult addRef();
}
