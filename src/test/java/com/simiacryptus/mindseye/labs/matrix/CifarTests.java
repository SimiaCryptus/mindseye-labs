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

import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer;
import com.simiacryptus.mindseye.layers.cudnn.conv.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.java.BiasLayer;
import com.simiacryptus.mindseye.layers.java.FullyConnectedLayer;
import com.simiacryptus.mindseye.layers.java.ReLuActivationLayer;
import com.simiacryptus.mindseye.layers.java.SoftmaxLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.data.CIFAR10;
import com.simiacryptus.mindseye.test.integration.*;
import com.simiacryptus.notebook.NotebookOutput;

import javax.annotation.Nonnull;

public class CifarTests {

  @Nonnull
  public static FwdNetworkFactory fwd_conv_1 = (log, features) -> {
    log.p("The png-to-vector network is a single key convolutional:");
    return log.eval(() -> {
      @Nonnull final PipelineNetwork network = new PipelineNetwork();
      ConvolutionLayer convolutionLayer = new ConvolutionLayer(3, 3, 3, 5);
      convolutionLayer.set(i -> 1e-8 * (Math.random() - 0.5));
      network.add(convolutionLayer.addRef());
      PoolingLayer poolingLayer = new PoolingLayer();
      poolingLayer.setMode(PoolingLayer.PoolingMode.Max);
      network.add(poolingLayer.addRef());
      network.add(new ReLuActivationLayer());
      network.add(new BiasLayer(16, 16, 5));
      FullyConnectedLayer fullyConnectedLayer = new FullyConnectedLayer(new int[]{16, 16, 5}, new int[]{features});
      fullyConnectedLayer.set(() -> 0.001 * (Math.random() - 0.45));
      network.add(fullyConnectedLayer.addRef());
      network.add(new SoftmaxLayer());
      return network;
    });
  };
  @Nonnull
  public static FwdNetworkFactory fwd_linear_1 = (log, features) -> {
    log.p("The png-to-vector network is a single key, fully connected:");
    return log.eval(() -> {
      @Nonnull final PipelineNetwork network = new PipelineNetwork();
      network.add(new BiasLayer(32, 32, 3));
      FullyConnectedLayer fullyConnectedLayer = new FullyConnectedLayer(new int[]{32, 32, 3}, new int[]{features});
      fullyConnectedLayer.set(() -> 0.001 * (Math.random() - 0.45));
      network.add(fullyConnectedLayer.addRef());
      network.add(new SoftmaxLayer());
      return network;
    });
  };
  @Nonnull
  public static RevNetworkFactory rev_conv_1 = (log, features) -> {
    log.p("The vector-to-png network uses a fully connected key then a single convolutional key:");
    return log.eval(() -> {
      @Nonnull final PipelineNetwork network = new PipelineNetwork();
      FullyConnectedLayer fullyConnectedLayer = new FullyConnectedLayer(new int[]{features}, new int[]{32, 32, 5});
      fullyConnectedLayer.set(() -> 0.25 * (Math.random() - 0.5));
      network.add(fullyConnectedLayer.addRef());
      network.add(new ReLuActivationLayer());
      ConvolutionLayer convolutionLayer = new ConvolutionLayer(3, 3, 5, 3);
      convolutionLayer.set(i -> 1e-8 * (Math.random() - 0.5));
      network.add(convolutionLayer.addRef());
      network.add(new BiasLayer(32, 32, 3));
      network.add(new ReLuActivationLayer());
      return network;
    });
  };
  @Nonnull
  public static RevNetworkFactory rev_linear_1 = (log, features) -> {
    log.p("The vector-to-png network is a single fully connected key:");
    return log.eval(() -> {
      @Nonnull final PipelineNetwork network = new PipelineNetwork();
      FullyConnectedLayer fullyConnectedLayer = new FullyConnectedLayer(new int[]{features}, new int[]{32, 32, 3});
      fullyConnectedLayer.set(() -> 0.25 * (Math.random() - 0.5));
      network.add(fullyConnectedLayer.addRef());
      network.add(new BiasLayer(32, 32, 3));
      return network;
    });
  };

  public abstract static class All_CIFAR_Tests extends AllTrainingTests {
    public All_CIFAR_Tests(final OptimizationStrategy optimizationStrategy, final RevNetworkFactory revFactory,
                           final FwdNetworkFactory fwdFactory) {
      super(fwdFactory, revFactory, optimizationStrategy);
    }

    @Nonnull
    @Override
    public ImageProblemData getData() {
      return new CIFARProblemData();
    }

    @Nonnull
    @Override
    public CharSequence getDatasetName() {
      return "CIFAR10";
    }

    @Nonnull
    @Override
    public ReportType getReportType() {
      return ReportType.Experiments;
    }

    @Nonnull
    @Override
    protected Class<?> getTargetClass() {
      return CIFAR10.class;
    }

  }

  public static class OWL_QN extends All_CIFAR_Tests {
    public OWL_QN() {
      super(TextbookOptimizers.orthantwise_quasi_newton, CifarTests.rev_conv_1, CifarTests.fwd_conv_1);
    }

    @Override
    protected void intro(@Nonnull final NotebookOutput log) {
      log.p("");
    }
  }

  public static class QQN extends All_CIFAR_Tests {
    public QQN() {
      super(Research.quadratic_quasi_newton, CifarTests.rev_conv_1, CifarTests.fwd_conv_1);
    }

    @Override
    protected void intro(@Nonnull final NotebookOutput log) {
      log.p("");
    }
  }

  public static class SGD extends All_CIFAR_Tests {
    public SGD() {
      super(TextbookOptimizers.stochastic_gradient_descent, CifarTests.rev_linear_1, CifarTests.fwd_linear_1);
    }

    @Override
    protected void intro(@Nonnull final NotebookOutput log) {
      log.p("");
    }
  }

}
