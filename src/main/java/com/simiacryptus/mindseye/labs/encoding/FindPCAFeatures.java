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

package com.simiacryptus.mindseye.labs.encoding;

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.test.PCAUtil;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefStream;
import org.apache.commons.math3.linear.RealMatrix;

import javax.annotation.Nonnull;
import java.util.function.Supplier;

abstract class FindPCAFeatures extends FindFeatureSpace {

  public FindPCAFeatures(final NotebookOutput log, final int inputBands) {
    super(log, inputBands);
  }

  @Nonnull
  protected abstract RefStream<Tensor[]> getFeatures();

  @Nonnull
  @Override
  public FindFeatureSpace invoke() {
    double[] averages = findBandBias();
    Tensor[] vectors = findFeatureSpace(log, () -> getFeatures().map(tensor -> {
      return new Tensor[]{tensor[0], tensor[1].mapCoords((c) -> tensor[1].get(c) - averages[c.getCoords()[2]])};
    }), inputBands);
    return this;
  }

  protected double[] findBandBias() {
    final int outputBands = RefUtil.get(getFeatures().findAny())[1].getDimensions()[2];
    return RefIntStream.range(0, outputBands).parallel().mapToDouble(b -> {
      return getFeatures().mapToDouble(tensor -> {
        return tensor[1].coordStream(false).filter((c) -> c.getCoords()[2] == b).mapToDouble((c) -> tensor[1].get(c))
            .average().getAsDouble();
      }).average().getAsDouble();
    }).toArray();
  }

  protected Tensor[] findFeatureSpace(@Nonnull final NotebookOutput log,
                                      @Nonnull final Supplier<RefStream<Tensor[]>> featureVectors, final int components) {
    return log.eval(() -> {
      final int column = 1;
      @Nonnull final Tensor[] prototype = RefUtil.get(featureVectors.get().findAny());
      @Nonnull final int[] dimensions = prototype[column].getDimensions();
      RealMatrix covariance = PCAUtil.getCovariance(() -> featureVectors.get().map(x -> x[column].getData()));
      return PCAUtil.pcaFeatures(covariance, components, dimensions, -1);
    });
  }

}
