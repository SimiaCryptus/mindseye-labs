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

import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.util.data.DoubleStatistics;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;

public class ToleranceStatistics {
  public final DoubleStatistics absoluteTol;
  public final DoubleStatistics relativeTol;

  public ToleranceStatistics() {
    this(new DoubleStatistics(), new DoubleStatistics());
  }

  public ToleranceStatistics(final DoubleStatistics absoluteTol, final DoubleStatistics relativeTol) {
    this.absoluteTol = absoluteTol;
    this.relativeTol = relativeTol;
  }

  @Nonnull
  public ToleranceStatistics accumulate(final double target, final double val) {
    absoluteTol.accept(Math.abs(target - val));
    if (Double.isFinite(val + target) && val != -target) {
      relativeTol.accept(Math.abs(target - val) / (Math.abs(val) + Math.abs(target)));
    }
    return this;
  }

  @Nonnull
  public ToleranceStatistics accumulate(@Nonnull final double[] target, @Nonnull final double[] val) {
    if (target.length != val.length)
      throw new IllegalArgumentException();
    RefIntStream.range(0, target.length).forEach(i -> accumulate(target[i], val[i]));
    return this;
  }

  @Nonnull
  public ToleranceStatistics combine(@Nullable final ToleranceStatistics right) {
    if (null == right)
      return this;
    return new ToleranceStatistics(absoluteTol.combine(right.absoluteTol), relativeTol.combine(right.relativeTol));
  }

  @Nonnull
  @Override
  public String toString() {
    return "ToleranceStatistics{" + "absoluteTol=" + absoluteTol + ", relativeTol=" + relativeTol + '}';
  }
}
