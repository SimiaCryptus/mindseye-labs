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

package com.simiacryptus.mindseye.opt.orient;

import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.SampledArrayTrainable;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.java.EntropyLossLayer;
import com.simiacryptus.mindseye.network.SimpleLossNetwork;
import com.simiacryptus.mindseye.opt.MnistTestBase;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.ValidatingTrainer;
import com.simiacryptus.mindseye.opt.line.QuadraticSearch;
import com.simiacryptus.notebook.NotebookOutput;

import javax.annotation.Nonnull;
import java.util.concurrent.TimeUnit;

public class LBFGSTest extends MnistTestBase {

  @Nonnull
  @Override
  protected Class<?> getTargetClass() {
    return LBFGS.class;
  }


  @Override
  public void train(@Nonnull final NotebookOutput log, @Nonnull final Layer network,
                    @Nonnull final Tensor[][] trainingData, final TrainingMonitor monitor) {
    log.eval(() -> {
      @Nonnull final SimpleLossNetwork supervisedNetwork = new SimpleLossNetwork(network, new EntropyLossLayer());
      ValidatingTrainer validatingTrainer1 = new ValidatingTrainer(
          new SampledArrayTrainable(trainingData, supervisedNetwork, 1000, 10000),
          new ArrayTrainable(trainingData, supervisedNetwork).cached());
      validatingTrainer1.setMonitor(monitor);
      @Nonnull
      ValidatingTrainer trainer = validatingTrainer1.addRef();
      //.setOrientation(new ValidatingOrientationWrapper(new LBFGS()))
      ValidatingTrainer.TrainingPhase trainingPhase = trainer.getRegimen().get(0);
      trainingPhase.setOrientation(new LBFGS());
      //.setOrientation(new ValidatingOrientationWrapper(new LBFGS()))
      trainingPhase.addRef().setLineSearchFactory(name -> name.toString().contains("LBFGS") ? new QuadraticSearch().setCurrentRate(1.0)
          : new QuadraticSearch());
      trainer.setTimeout(5, TimeUnit.MINUTES);
      ValidatingTrainer validatingTrainer = trainer.addRef();
      validatingTrainer.setMaxIterations(500);
      return validatingTrainer.addRef().run();
    });
  }

}
