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

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.ref.lang.RecycleBin;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.data.DoubleStatistics;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.RealMatrix;

import javax.annotation.Nonnull;
import java.util.function.Supplier;

public class PCAUtil {
  @Nonnull
  public static RealMatrix getCovariance(@Nonnull final Supplier<RefStream<double[]>> stream) {
    final int dimension = RefUtil.get(stream.get().findAny()).length;
    final RefList<DoubleStatistics> statList = RefIntStream.range(0, dimension * dimension)
        .mapToObj(i -> new DoubleStatistics()).collect(RefCollectors.toList());
    stream.get().forEach(array -> {
      for (int i = 0; i < dimension; i++) {
        for (int j = 0; j <= i; j++) {
          statList.get(i * dimension + j).accept(array[i] * array[j]);
        }
      }
      RecycleBin.DOUBLES.recycle(array, array.length);
    });
    @Nonnull final RealMatrix covariance = new BlockRealMatrix(dimension, dimension);
    for (int i = 0; i < dimension; i++) {
      for (int j = 0; j <= i; j++) {
        final double v = statList.get(i + dimension * j).getAverage();
        covariance.setEntry(i, j, v);
        covariance.setEntry(j, i, v);
      }
    }
    return covariance;
  }

  @Nonnull
  public static Tensor[] pcaFeatures(@Nonnull final RealMatrix covariance, final int components, final int[] featureDimensions,
                                     final double power) {
    @Nonnull final EigenDecomposition decomposition = new EigenDecomposition(covariance);
    final int[] orderedVectors = RefIntStream.range(0, components).mapToObj(x -> x)
        .sorted(RefComparator.comparingDouble(x -> -decomposition.getRealEigenvalue(x))).mapToInt(x -> x).toArray();
    return RefIntStream.range(0, orderedVectors.length).mapToObj(i -> {
      @Nonnull final Tensor src = new Tensor(decomposition.getEigenvector(orderedVectors[i]).toArray(), featureDimensions)
          .copy();
      return src.scale(1.0 / src.rms())
          .scale(Math.pow(
              decomposition.getRealEigenvalue(orderedVectors[i]) / decomposition.getRealEigenvalue(orderedVectors[0]),
              power));
    }).toArray(i -> new Tensor[i]);
  }

  public static void populatePCAKernel_1(@Nonnull final Tensor kernel, @Nonnull final Tensor[] featureSpaceVectors) {
    final int outputBands = featureSpaceVectors.length;
    @Nonnull final int[] filterDimensions = kernel.getDimensions();
    kernel.setByCoord(c -> {
      final int kband = c.getCoords()[2];
      final int outband = kband % outputBands;
      final int inband = (kband - outband) / outputBands;
      int x = c.getCoords()[0];
      int y = c.getCoords()[1];
      x = filterDimensions[0] - (x + 1);
      y = filterDimensions[1] - (y + 1);
      final double v = featureSpaceVectors[outband].get(x, y, inband);
      return Double.isFinite(v) ? v : kernel.get(c);
    });
  }

  public static void populatePCAKernel_2(@Nonnull final Tensor kernel, @Nonnull final Tensor[] featureSpaceVectors) {
    final int outputBands = featureSpaceVectors.length;
    @Nonnull final int[] filterDimensions = kernel.getDimensions();
    kernel.setByCoord(c -> {
      final int kband = c.getCoords()[2];
      final int outband = kband % outputBands;
      final int inband = (kband - outband) / outputBands;
      int x = c.getCoords()[0];
      int y = c.getCoords()[1];
      x = filterDimensions[0] - (x + 1);
      y = filterDimensions[1] - (y + 1);
      final double v = featureSpaceVectors[inband].get(x, y, outband);
      return Double.isFinite(v) ? v : kernel.get(c);
    });
  }
}
