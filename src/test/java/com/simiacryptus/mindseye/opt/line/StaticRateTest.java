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

package com.simiacryptus.mindseye.opt.line;

import com.simiacryptus.mindseye.eval.SampledArrayTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.java.EntropyLossLayer;
import com.simiacryptus.mindseye.network.SimpleLossNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.MnistTestBase;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.orient.GradientDescent;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefUtil;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.concurrent.TimeUnit;

public class StaticRateTest extends MnistTestBase {

  @Nonnull
  @Override
  protected Class<?> getTargetClass() {
    return StaticLearningRate.class;
  }

  @Nullable
  public static @SuppressWarnings("unused")
  StaticRateTest[] addRefs(@Nullable StaticRateTest[] array) {
    if (array == null)
      return null;
    return Arrays.stream(array).filter(x -> x != null)
        .toArray(x -> new StaticRateTest[x]);
  }

  @Nullable
  public static @SuppressWarnings("unused")
  StaticRateTest[][] addRefs(@Nullable StaticRateTest[][] array) {
    return RefUtil.addRefs(array);
  }

  @Override
  public void train(@Nonnull final NotebookOutput log, @Nonnull final Layer network,
                    @Nonnull final Tensor[][] trainingData, final TrainingMonitor monitor) {
    log.eval(() -> {
      @Nonnull final SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
      @Nonnull final Trainable trainable = new SampledArrayTrainable(trainingData, supervisedNetwork, 1000);
      IterativeTrainer iterativeTrainer2 = new IterativeTrainer(trainable);
      iterativeTrainer2.setMonitor(monitor);
      IterativeTrainer iterativeTrainer3 = iterativeTrainer2.addRef();
      iterativeTrainer3.setOrientation(new GradientDescent());
      IterativeTrainer iterativeTrainer = iterativeTrainer3.addRef();
      iterativeTrainer.setLineSearchFactory((@Nonnull final CharSequence name) -> new StaticLearningRate(0.001));
      IterativeTrainer iterativeTrainer4 = iterativeTrainer.addRef();
      iterativeTrainer4.setTimeout(3, TimeUnit.MINUTES);
      IterativeTrainer iterativeTrainer1 = iterativeTrainer4.addRef();
      iterativeTrainer1.setMaxIterations(500);
      return iterativeTrainer1.addRef().run();
    });
  }

}
