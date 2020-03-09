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

package com.simiacryptus.mindseye.labs.matrix;

import com.simiacryptus.mindseye.opt.ValidatingTrainer;
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch;
import com.simiacryptus.mindseye.opt.line.QuadraticSearch;
import com.simiacryptus.mindseye.opt.line.StaticLearningRate;
import com.simiacryptus.mindseye.opt.orient.GradientDescent;
import com.simiacryptus.mindseye.opt.orient.LBFGS;
import com.simiacryptus.mindseye.opt.orient.MomentumStrategy;
import com.simiacryptus.mindseye.opt.orient.OwlQn;
import com.simiacryptus.mindseye.test.ProblemRun;
import com.simiacryptus.mindseye.test.StepRecord;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.test.integration.MnistProblemData;
import com.simiacryptus.mindseye.test.integration.OptimizationStrategy;
import com.simiacryptus.notebook.NotebookOutput;

import javax.annotation.Nonnull;
import java.awt.*;
import java.util.List;
import java.util.function.Function;

public class TextbookOptimizers extends OptimizerComparison {

  @Nonnull
  public static OptimizationStrategy conjugate_gradient_descent = (log, trainingSubject, validationSubject,
                                                                   monitor) -> {
    log.p("Optimized via the Conjugate Gradient Descent method:");
    return log.eval(() -> {
      ValidatingTrainer validatingTrainer = new ValidatingTrainer(trainingSubject, validationSubject);
      validatingTrainer.setMinTrainingSize(Integer.MAX_VALUE);
      validatingTrainer.setMonitor(monitor);
      ValidatingTrainer.TrainingPhase trainingPhase = validatingTrainer.getRegimen().get(0);
      trainingPhase.setOrientation(new GradientDescent());
      trainingPhase.setLineSearchFactory(name -> new QuadraticSearch().setRelativeTolerance(1e-5));
      return validatingTrainer;
    });
  };
  @Nonnull
  public static OptimizationStrategy limited_memory_bfgs = (log, trainingSubject, validationSubject, monitor) -> {
    log.p("Optimized via the Limited-Memory BFGS method:");
    return log.eval(() -> {
      ValidatingTrainer validatingTrainer = new ValidatingTrainer(trainingSubject, validationSubject);
      validatingTrainer.setMinTrainingSize(Integer.MAX_VALUE);
      validatingTrainer.setMonitor(monitor);
      ValidatingTrainer.TrainingPhase trainingPhase = validatingTrainer.getRegimen().get(0);
      trainingPhase.setOrientation(new LBFGS());
      trainingPhase.setLineSearchFactory(name -> new ArmijoWolfeSearch().setAlpha(name.toString().contains("LBFGS") ? 1.0 : 1e-6));
      return validatingTrainer;
    });
  };
  @Nonnull
  public static OptimizationStrategy orthantwise_quasi_newton = (log, trainingSubject, validationSubject, monitor) -> {
    log.p("Optimized via the Orthantwise Quasi-Newton search method:");
    return log.eval(() -> {
      ValidatingTrainer validatingTrainer = new ValidatingTrainer(trainingSubject, validationSubject);
      validatingTrainer.setMinTrainingSize(Integer.MAX_VALUE);
      validatingTrainer.setMonitor(monitor);
      ValidatingTrainer.TrainingPhase trainingPhase = validatingTrainer.getRegimen().get(0);
      trainingPhase.setOrientation(new OwlQn());
      trainingPhase.setLineSearchFactory(name -> new ArmijoWolfeSearch().setAlpha(name.toString().contains("OWL") ? 1.0 : 1e-6));
      return validatingTrainer;
    });
  };
  @Nonnull
  public static OptimizationStrategy simple_gradient_descent = (log, trainingSubject, validationSubject, monitor) -> {
    log.p("Optimized via the Stochastic Gradient Descent method:");
    return log.eval(() -> {
      final double rate = 0.05;
      ValidatingTrainer validatingTrainer1 = new ValidatingTrainer(trainingSubject, validationSubject);
      validatingTrainer1.setMinTrainingSize(Integer.MAX_VALUE);
      validatingTrainer1.setMaxEpochIterations(100);
      validatingTrainer1.setMonitor(monitor);
      ValidatingTrainer.TrainingPhase trainingPhase = validatingTrainer1.getRegimen().get(0);
      trainingPhase.setOrientation(new GradientDescent());
      trainingPhase.setLineSearchFactory(name -> new StaticLearningRate(rate));
      return validatingTrainer1;
    });
  };
  @Nonnull
  public static OptimizationStrategy stochastic_gradient_descent = (log, trainingSubject, validationSubject,
                                                                    monitor) -> {
    log.p("Optimized via the Stochastic Gradient Descent method apply momentum and adaptve learning rate:");
    return log.eval(() -> {
      final double carryOver = 0.5;
      ValidatingTrainer validatingTrainer = new ValidatingTrainer(trainingSubject, validationSubject);
      validatingTrainer.setMaxEpochIterations(100);
      validatingTrainer.setMonitor(monitor);
      ValidatingTrainer.TrainingPhase trainingPhase = validatingTrainer.getRegimen().get(0);
      MomentumStrategy momentumStrategy = new MomentumStrategy(new GradientDescent());
      momentumStrategy.setCarryOver(carryOver);
      trainingPhase.setOrientation(momentumStrategy.addRef());
      trainingPhase.setLineSearchFactory(name -> new ArmijoWolfeSearch());
      return validatingTrainer;
    });
  };

  public TextbookOptimizers() {
    super(MnistTests.fwd_conv_1, MnistTests.rev_conv_1, new MnistProblemData());
  }

  @Override
  public void compare(@Nonnull final NotebookOutput log,
                      @Nonnull final Function<OptimizationStrategy, List<StepRecord>> test) {
    log.h1("Textbook Optimizer Comparison");
    log.h2("GD");
    @Nonnull final ProblemRun gd = new ProblemRun("GD", test.apply(TextbookOptimizers.simple_gradient_descent), Color.BLACK,
        ProblemRun.PlotType.Line);
    log.h2("SGD");
    @Nonnull final ProblemRun sgd = new ProblemRun("SGD", test.apply(TextbookOptimizers.stochastic_gradient_descent),
        Color.GREEN, ProblemRun.PlotType.Line);
    log.h2("CGD");
    @Nonnull final ProblemRun cgd = new ProblemRun("CjGD", test.apply(TextbookOptimizers.conjugate_gradient_descent), Color.BLUE,
        ProblemRun.PlotType.Line);
    log.h2("L-BFGS");
    @Nonnull final ProblemRun lbfgs = new ProblemRun("L-BFGS", test.apply(TextbookOptimizers.limited_memory_bfgs), Color.MAGENTA,
        ProblemRun.PlotType.Line);
    log.h2("OWL-QN");
    @Nonnull final ProblemRun owlqn = new ProblemRun("OWL-QN", test.apply(TextbookOptimizers.orthantwise_quasi_newton),
        Color.ORANGE, ProblemRun.PlotType.Line);
    log.h2("Comparison");
    log.eval(() -> {
      return TestUtil.compare("Convergence Plot", gd, sgd, cgd, lbfgs, owlqn);
    });
    log.eval(() -> {
      return TestUtil.compareTime("Convergence Plot", gd, sgd, cgd, lbfgs, owlqn);
    });
  }

}
