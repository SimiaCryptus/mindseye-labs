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

package com.simiacryptus.mindseye.network;

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.layers.java.BiasLayer;
import com.simiacryptus.mindseye.layers.java.FullyConnectedLayer;
import com.simiacryptus.mindseye.layers.java.ReLuActivationLayer;
import com.simiacryptus.mindseye.layers.java.SigmoidActivationLayer;
import com.simiacryptus.util.Util;

import javax.annotation.Nonnull;

public abstract class DeepLinear extends NLayerTest {

  public DeepLinear(final int[]... dimList) {
    super(dimList);
  }

  @Override
  public void addLayer(@Nonnull final PipelineNetwork network, @Nonnull final int[] in, @Nonnull final int[] dims) {
    network.wrap(new FullyConnectedLayer(in, dims).set(this::random)).freeRef();
    network.wrap(new BiasLayer(dims)).freeRef();
    network.wrap(getActivation()).freeRef();
  }

  @Nonnull
  public Layer getActivation() {
    return new ReLuActivationLayer();
  }

  @Nonnull
  @Override
  public int[] getInputDims() {
    return new int[]{5, 5, 3};
  }

  @Override
  public double random() {
    return 0.1 * Math.round(1000.0 * (Util.R.get().nextDouble() - 0.5)) / 500.0;
  }

  public static class NarrowingPipeline extends DeepLinear {
    public NarrowingPipeline() {
      super(
          new int[]{4, 4, 2},
          new int[]{3, 3, 1},
          new int[]{2, 2, 1},
          new int[]{2, 2, 1}
      );
    }

  }

  public static class SigmoidPipeline extends DeepLinear {
    public SigmoidPipeline() {
      super(
          new int[]{10},
          new int[]{10},
          new int[]{10},
          new int[]{10}
      );
    }

    @Nonnull
    @Override
    public Layer getActivation() {
      return new SigmoidActivationLayer();
    }

  }

  public static class UniformPipeline extends DeepLinear {
    public UniformPipeline() {
      super(
          new int[]{10},
          new int[]{10},
          new int[]{10},
          new int[]{10}
      );
    }

  }

}
