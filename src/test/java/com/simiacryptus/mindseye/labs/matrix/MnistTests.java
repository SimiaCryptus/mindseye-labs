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
import com.simiacryptus.mindseye.layers.java.*;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.data.MNIST;
import com.simiacryptus.mindseye.test.integration.*;
import com.simiacryptus.notebook.NotebookOutput;

import javax.annotation.Nonnull;
import java.util.function.DoubleSupplier;

public class MnistTests {
  @Nonnull
  public static FwdNetworkFactory fwd_conv_1 = (log, features) -> {
    log.p("The png-to-vector network is a single key convolutional:");
    return log.eval(() -> {
      @Nonnull final PipelineNetwork network = new PipelineNetwork();
      ConvolutionLayer convolutionLayer1 = new ConvolutionLayer(5, 5, 1, 32);
      convolutionLayer1.set(i1 -> 1e-8 * (Math.random() - 0.5));
      network.add(convolutionLayer1.addRef());
      network.add(new ImgBandBiasLayer(32));
      PoolingLayer poolingLayer1 = new PoolingLayer();
      poolingLayer1.setMode(PoolingLayer.PoolingMode.Max);
      network.add(poolingLayer1.addRef());
      ConvolutionLayer convolutionLayer = new ConvolutionLayer(5, 5, 32, 64);
      convolutionLayer.set(i -> 1e-8 * (Math.random() - 0.5));
      network.add(convolutionLayer.addRef());
      network.add(new ImgBandBiasLayer(64));
      PoolingLayer poolingLayer = new PoolingLayer();
      poolingLayer.setMode(PoolingLayer.PoolingMode.Max);
      network.add(poolingLayer.addRef());
      network.add(new ReLuActivationLayer());
      FullyConnectedLayer fullyConnectedLayer1 = new FullyConnectedLayer(new int[]{7, 7, 64}, new int[]{1024});
      fullyConnectedLayer1.set(() -> 0.001 * (Math.random() - 0.45));
      network.add(fullyConnectedLayer1.addRef());
      network.add(new BiasLayer(1024));
      network.add(new ReLuActivationLayer());
      network.add(new DropoutNoiseLayer(0.5));
      FullyConnectedLayer fullyConnectedLayer = new FullyConnectedLayer(new int[]{1024}, new int[]{features});
      fullyConnectedLayer.set(() -> 0.001 * (Math.random() - 0.45));
      network.add(fullyConnectedLayer.addRef());
      network.add(new BiasLayer(features));
      network.add(new SoftmaxLayer());
      return network;
    });
  };

  @Nonnull
  public static FwdNetworkFactory fwd_conv_1_n = (log, features) -> {
    log.p("The png-to-vector network is a single key convolutional:");
    return log.eval(() -> {
      @Nonnull final PipelineNetwork network = new PipelineNetwork();
      double weight = 1e-3;

      network.add(new NormalizationMetaLayer());
      @Nonnull
      DoubleSupplier init = () -> weight * (Math.random() - 0.5);

      ConvolutionLayer convolutionLayer1 = new ConvolutionLayer(5, 5, 1, 32);
      convolutionLayer1.set(init);
      network.add(convolutionLayer1);
      network.add(new ImgBandBiasLayer(32));
      network.add(new NormalizationMetaLayer());
      PoolingLayer poolingLayer1 = new PoolingLayer();
      poolingLayer1.setMode(PoolingLayer.PoolingMode.Max);
      network.add(poolingLayer1.addRef());
      ConvolutionLayer convolutionLayer = new ConvolutionLayer(5, 5, 32, 64);
      convolutionLayer.set(init);
      network.add(convolutionLayer);
      network.add(new ImgBandBiasLayer(64));
      PoolingLayer poolingLayer = new PoolingLayer();
      poolingLayer.setMode(PoolingLayer.PoolingMode.Max);
      network.add(poolingLayer.addRef());
      network.add(new ReLuActivationLayer());
      network.add(new NormalizationMetaLayer());
      FullyConnectedLayer fullyConnectedLayer1 = new FullyConnectedLayer(new int[]{4, 4, 64}, new int[]{1024});
      fullyConnectedLayer1.set(init);
      network.add(fullyConnectedLayer1.addRef());
      network.add(new BiasLayer(1024));
      network.add(new ReLuActivationLayer());
      network.add(new NormalizationMetaLayer());
      network.add(new DropoutNoiseLayer(0.5));
      FullyConnectedLayer fullyConnectedLayer = new FullyConnectedLayer(new int[]{1024}, new int[]{features});
      fullyConnectedLayer.set(init);
      network.add(fullyConnectedLayer.addRef());
      network.add(new BiasLayer(features));
      network.add(new SoftmaxLayer());

      return network;
    });
  };

  @Nonnull
  public static FwdNetworkFactory fwd_linear_1 = (log, features) -> {
    log.p("The png-to-vector network is a single key, fully connected:");
    return log.eval(() -> {
      @Nonnull final PipelineNetwork network = new PipelineNetwork();
      network.add(new BiasLayer(28, 28, 1));
      FullyConnectedLayer fullyConnectedLayer = new FullyConnectedLayer(new int[]{28, 28, 1}, new int[]{features});
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
      FullyConnectedLayer fullyConnectedLayer1 = new FullyConnectedLayer(new int[]{features}, new int[]{1024});
      fullyConnectedLayer1.set(() -> 0.25 * (Math.random() - 0.5));
      network.add(
          fullyConnectedLayer1.addRef());
      network.add(new DropoutNoiseLayer(0.5));
      network.add(new ReLuActivationLayer());
      network.add(new BiasLayer(1024));
      FullyConnectedLayer fullyConnectedLayer = new FullyConnectedLayer(new int[]{1024}, new int[]{4, 4, 64});
      fullyConnectedLayer.set(() -> 0.001 * (Math.random() - 0.45));
      network.add(fullyConnectedLayer.addRef());
      network.add(new ReLuActivationLayer());

      ConvolutionLayer convolutionLayer3 = new ConvolutionLayer(1, 1, 64, 4 * 64);
      convolutionLayer3.set(i3 -> 1e-8 * (Math.random() - 0.5));
      network.add(convolutionLayer3.addRef());
      network.add(new ImgReshapeLayer(2, 2, true));
      network.add(new ImgBandBiasLayer(64));
      ConvolutionLayer convolutionLayer2 = new ConvolutionLayer(5, 5, 64, 32);
      convolutionLayer2.set(i2 -> 1e-8 * (Math.random() - 0.5));
      network.add(convolutionLayer2.addRef());

      ConvolutionLayer convolutionLayer1 = new ConvolutionLayer(1, 1, 32, 4 * 32);
      convolutionLayer1.set(i1 -> 1e-8 * (Math.random() - 0.5));
      network.add(convolutionLayer1.addRef());
      network.add(new ImgReshapeLayer(2, 2, true));
      network.add(new ImgBandBiasLayer(32));
      ConvolutionLayer convolutionLayer = new ConvolutionLayer(5, 5, 32, 1);
      convolutionLayer.set(i -> 1e-8 * (Math.random() - 0.5));
      network.add(convolutionLayer.addRef());

      return network;
    });
  };
  @Nonnull
  public static RevNetworkFactory rev_linear_1 = (log, features) -> {
    log.p("The vector-to-png network is a single fully connected key:");
    return log.eval(() -> {
      @Nonnull final PipelineNetwork network = new PipelineNetwork();
      FullyConnectedLayer fullyConnectedLayer = new FullyConnectedLayer(new int[]{features}, new int[]{28, 28, 1});
      fullyConnectedLayer.set(() -> 0.25 * (Math.random() - 0.5));
      network.add(fullyConnectedLayer.addRef());
      network.add(new BiasLayer(28, 28, 1));
      return network;
    });
  };

  public abstract static class All_MNIST_Tests extends AllTrainingTests {
    public All_MNIST_Tests(final OptimizationStrategy optimizationStrategy, final RevNetworkFactory revFactory,
                           final FwdNetworkFactory fwdFactory) {
      super(fwdFactory, revFactory, optimizationStrategy);
    }

    @Nonnull
    @Override
    public ImageProblemData getData() {
      return new MnistProblemData();
    }

    @Nonnull
    @Override
    public CharSequence getDatasetName() {
      return "MNIST";
    }

    @Nonnull
    @Override
    public ReportType getReportType() {
      return ReportType.Experiments;
    }

    @Nonnull
    @Override
    protected Class<?> getTargetClass() {
      return MNIST.class;
    }

  }

  public static class OWL_QN extends All_MNIST_Tests {
    public OWL_QN() {
      super(TextbookOptimizers.orthantwise_quasi_newton, MnistTests.rev_conv_1, MnistTests.fwd_conv_1);
    }

    @Override
    protected void intro(@Nonnull final NotebookOutput log) {
      log.p("");
    }
  }

  public static class QQN extends All_MNIST_Tests {
    public QQN() {
      super(Research.quadratic_quasi_newton, MnistTests.rev_conv_1, MnistTests.fwd_conv_1);
    }

    @Override
    protected void intro(@Nonnull final NotebookOutput log) {
      log.p("");
    }
  }

  public static class SGD extends All_MNIST_Tests {
    public SGD() {
      super(TextbookOptimizers.stochastic_gradient_descent, MnistTests.rev_linear_1, MnistTests.fwd_linear_1);
    }

    @Override
    protected void intro(@Nonnull final NotebookOutput log) {
      log.p("");
    }
  }

}
