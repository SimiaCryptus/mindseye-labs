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

package com.simiacryptus.mindseye.opt.region;

import com.simiacryptus.mindseye.eval.SampledArrayTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.java.EntropyLossLayer;
import com.simiacryptus.mindseye.network.SimpleLossNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.MnistTestBase;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.orient.TrustRegionStrategy;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefUtil;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;

public class SingleOrthantTrustRegionTest extends MnistTestBase {

  @Nonnull
  @Override
  protected Class<?> getTargetClass() {
    return SingleOrthant.class;
  }

  @Nullable
  public static @SuppressWarnings("unused")
  SingleOrthantTrustRegionTest[] addRef(
      @Nullable SingleOrthantTrustRegionTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter(x -> x != null)
        .toArray(x -> new SingleOrthantTrustRegionTest[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  SingleOrthantTrustRegionTest[][] addRef(
      @Nullable SingleOrthantTrustRegionTest[][] array) {
    return RefUtil.addRef(array);
  }

  @Override
  public void train(@Nonnull final NotebookOutput log, @Nonnull final Layer network,
                    @Nonnull final Tensor[][] trainingData, final TrainingMonitor monitor) {
    log.eval(() -> {
      @Nonnull final SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
      @Nonnull final Trainable trainable = new SampledArrayTrainable(trainingData, supervisedNetwork, 10000);
      @Nonnull final TrustRegionStrategy trustRegionStrategy = new TrustRegionStrategy() {
        @Nonnull
        @Override
        public TrustRegion getRegionPolicy(final Layer layer) {
          if (null != layer) layer.freeRef();
          return new SingleOrthant();
        }

        public @SuppressWarnings("unused")
        void _free() {
        }
      };
      //.setOrientation(new ValidatingOrientationWrapper(trustRegionStrategy))
      IterativeTrainer iterativeTrainer = new IterativeTrainer(trainable);
      iterativeTrainer.setIterationsPerSample(100);
      IterativeTrainer iterativeTrainer2 = iterativeTrainer.addRef();
      iterativeTrainer2.setMonitor(monitor);
      IterativeTrainer iterativeTrainer3 = iterativeTrainer2.addRef();
      iterativeTrainer3.setOrientation(trustRegionStrategy);
      //.setOrientation(new ValidatingOrientationWrapper(trustRegionStrategy))
      IterativeTrainer iterativeTrainer4 = iterativeTrainer3.addRef();
      iterativeTrainer4.setTimeout(3, TimeUnit.MINUTES);
      IterativeTrainer iterativeTrainer1 = iterativeTrainer4.addRef();
      iterativeTrainer1.setMaxIterations(500);
      //.setOrientation(new ValidatingOrientationWrapper(trustRegionStrategy))
      return iterativeTrainer1.addRef().run();
    });
  }

}
