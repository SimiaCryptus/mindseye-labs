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

import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.BasicTrainable;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.ValidatingTrainer;
import com.simiacryptus.mindseye.opt.line.QuadraticSearch;
import com.simiacryptus.mindseye.opt.line.StaticLearningRate;
import com.simiacryptus.mindseye.opt.orient.LBFGS;
import com.simiacryptus.mindseye.opt.orient.QQN;
import com.simiacryptus.mindseye.opt.orient.RecursiveSubspace;
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

public class Research extends OptimizerComparison {

  @Nonnull
  public static OptimizationStrategy recursive_subspace = (log, trainingSubject, validationSubject, monitor) -> {
    log.p("Optimized via the Recursive Subspace method:");
    return log.eval(() -> {
      ValidatingTrainer validatingTrainer = new ValidatingTrainer(trainingSubject, validationSubject);
      validatingTrainer.setMonitor(monitor);
      @Nonnull final ValidatingTrainer trainer = validatingTrainer.addRef();
      //new SingleDerivativeTester(1e-3,1e-4).apply(subspace, new Tensor[]{new Tensor()});
      ValidatingTrainer.TrainingPhase trainingPhase = trainer.getRegimen().get(0);
      trainingPhase.setOrientation(new RecursiveSubspace() {
        @Override
        public void train(@Nonnull TrainingMonitor monitor1, Layer subspace) {
          //new SingleDerivativeTester(1e-3,1e-4).apply(subspace, new Tensor[]{new Tensor()});
          super.train(monitor1, subspace);
        }

        public @SuppressWarnings("unused")
        void _free() {
        }
      });
      //new SingleDerivativeTester(1e-3,1e-4).apply(subspace, new Tensor[]{new Tensor()});
      trainingPhase.setLineSearchFactory(name -> new StaticLearningRate(1.0));
      return trainer;
    });
  };
  @Nonnull
  public static OptimizationStrategy recursive_subspace_2 = (log, trainingSubject, validationSubject, monitor) -> {
    log.p("Optimized via the Recursive Subspace method:");
    return log.eval(() -> {
      ValidatingTrainer validatingTrainer = new ValidatingTrainer(trainingSubject, validationSubject);
      validatingTrainer.setMonitor(monitor);
      @Nonnull final ValidatingTrainer trainer = validatingTrainer.addRef();
      //new SingleDerivativeTester(1e-3,1e-4).apply(subspace, new Tensor[]{new Tensor()});
      ValidatingTrainer.TrainingPhase trainingPhase = trainer.getRegimen().get(0);
      trainingPhase.setOrientation(new RecursiveSubspace() {
        @Override
        public void train(@Nonnull TrainingMonitor monitor1, Layer subspace) {
          //new SingleDerivativeTester(1e-3,1e-4).apply(subspace, new Tensor[]{new Tensor()});
          @Nonnull
          ArrayTrainable trainable = new ArrayTrainable(new BasicTrainable(subspace),
              new Tensor[][]{{new Tensor()}});
          IterativeTrainer iterativeTrainer4 = new IterativeTrainer(trainable);
          iterativeTrainer4.setOrientation(new QQN());
          IterativeTrainer iterativeTrainer1 = iterativeTrainer4.addRef();
          iterativeTrainer1.setLineSearchFactory(n -> new QuadraticSearch());
          IterativeTrainer iterativeTrainer3 = iterativeTrainer1.addRef();
          iterativeTrainer3.setMonitor(new TrainingMonitor() {
            @Override
            public void log(String msg) {
              monitor1.log("\t" + msg);
            }
          });
          IterativeTrainer iterativeTrainer2 = iterativeTrainer3.addRef();
          iterativeTrainer2.setMaxIterations(getIterations());
          IterativeTrainer iterativeTrainer = iterativeTrainer2.addRef();
          iterativeTrainer.setIterationsPerSample(getIterations());
          iterativeTrainer.addRef().run();
        }

        public @SuppressWarnings("unused")
        void _free() {
        }
      });
      //new SingleDerivativeTester(1e-3,1e-4).apply(subspace, new Tensor[]{new Tensor()});
      trainingPhase.setLineSearchFactory(name -> new StaticLearningRate(1.0));
      return trainer;
    });
  };

  @Nonnull
  public static OptimizationStrategy quadratic_quasi_newton = (log, trainingSubject, validationSubject, monitor) -> {
    log.p("Optimized via the Quadratic Quasi-Newton method:");
    return log.eval(() -> {
      ValidatingTrainer validatingTrainer = new ValidatingTrainer(trainingSubject, validationSubject);
      validatingTrainer.setMonitor(monitor);
      ValidatingTrainer.TrainingPhase trainingPhase = validatingTrainer.getRegimen().get(0);
      trainingPhase.setOrientation(new QQN());
      trainingPhase.setLineSearchFactory(name -> new QuadraticSearch()
          .setCurrentRate(name.toString().contains("QQN") ? 1.0 : 1e-6).setRelativeTolerance(2e-1));
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
      trainingPhase.setLineSearchFactory(name -> new QuadraticSearch().setCurrentRate(name.toString().contains("LBFGS") ? 1.0 : 1e-6));
      return validatingTrainer;
    });
  };

  public Research() {
    super(MnistTests.fwd_conv_1, MnistTests.rev_conv_1, new MnistProblemData());
  }


  @Override
  public void compare(@Nonnull final NotebookOutput log,
                      @Nonnull final Function<OptimizationStrategy, List<StepRecord>> test) {
    log.h1("Research Optimizer Comparison");

    log.h2("Recursive Subspace (Un-Normalized)");
    fwdFactory = MnistTests.fwd_conv_1;
    @Nonnull final ProblemRun subspace_1 = new ProblemRun("SS", test.apply(Research.recursive_subspace), Color.LIGHT_GRAY,
        ProblemRun.PlotType.Line);

    log.h2("Recursive Subspace (Un-Normalized)");
    fwdFactory = MnistTests.fwd_conv_1;
    @Nonnull final ProblemRun subspace_2 = new ProblemRun("SS+QQN", test.apply(Research.recursive_subspace_2), Color.RED,
        ProblemRun.PlotType.Line);

    log.h2("QQN (Normalized)");
    fwdFactory = MnistTests.fwd_conv_1_n;
    @Nonnull final ProblemRun qqn1 = new ProblemRun("QQN", test.apply(Research.quadratic_quasi_newton), Color.DARK_GRAY,
        ProblemRun.PlotType.Line);

    log.h2("L-BFGS (Strong Line Search) (Normalized)");
    fwdFactory = MnistTests.fwd_conv_1_n;
    @Nonnull final ProblemRun lbfgs_2 = new ProblemRun("LB-2", test.apply(Research.limited_memory_bfgs), Color.MAGENTA,
        ProblemRun.PlotType.Line);

    log.h2("L-BFGS (Normalized)");
    fwdFactory = MnistTests.fwd_conv_1_n;
    @Nonnull final ProblemRun lbfgs_1 = new ProblemRun("LB-1", test.apply(TextbookOptimizers.limited_memory_bfgs), Color.GREEN,
        ProblemRun.PlotType.Line);

    log.h2("L-BFGS-0 (Un-Normalized)");
    fwdFactory = MnistTests.fwd_conv_1;
    @Nonnull final ProblemRun rawlbfgs = new ProblemRun("LBFGS-0", test.apply(TextbookOptimizers.limited_memory_bfgs),
        Color.CYAN, ProblemRun.PlotType.Line);

    log.h2("Comparison");
    log.eval(() -> {
      return TestUtil.compare("Convergence Plot", subspace_1, subspace_2, rawlbfgs, lbfgs_1, lbfgs_2, qqn1);
    });
    log.eval(() -> {
      return TestUtil.compareTime("Convergence Plot", subspace_1, subspace_2, rawlbfgs, lbfgs_1, lbfgs_2, qqn1);
    });
  }

}
