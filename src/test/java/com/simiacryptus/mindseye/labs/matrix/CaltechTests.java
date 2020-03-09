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
import com.simiacryptus.mindseye.test.data.Caltech101;
import com.simiacryptus.mindseye.test.integration.*;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.lang.RefUtil;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.function.IntToDoubleFunction;

public class CaltechTests {

  @Nonnull
  public static FwdNetworkFactory fwd_conv_1 = (log, features) -> {
    log.p("The png-to-vector network is a single key convolutional:");
    return log.eval(() -> {
      @Nonnull final PipelineNetwork network = new PipelineNetwork();

      @Nonnull
      IntToDoubleFunction weights = i -> 1e-8 * (Math.random() - 0.5);
      ConvolutionLayer convolutionLayer3 = new ConvolutionLayer(3, 3, 3, 10);
      convolutionLayer3.set(weights);
      network.add(convolutionLayer3.addRef()).freeRef();
      PoolingLayer poolingLayer3 = new PoolingLayer();
      poolingLayer3.setMode(PoolingLayer.PoolingMode.Max);
      network.add(poolingLayer3.addRef()).freeRef();
      network.add(new ReLuActivationLayer()).freeRef();
      network.add(new ImgCropLayer(126, 126)).freeRef();
      network.add(new NormalizationMetaLayer()).freeRef();

      ConvolutionLayer convolutionLayer2 = new ConvolutionLayer(3, 3, 10, 20);
      convolutionLayer2.set(weights);
      network.add(convolutionLayer2.addRef()).freeRef();
      PoolingLayer poolingLayer2 = new PoolingLayer();
      poolingLayer2.setMode(PoolingLayer.PoolingMode.Max);
      network.add(poolingLayer2.addRef()).freeRef();
      network.add(new ReLuActivationLayer()).freeRef();
      network.add(new ImgCropLayer(62, 62)).freeRef();
      network.add(new NormalizationMetaLayer()).freeRef();

      ConvolutionLayer convolutionLayer1 = new ConvolutionLayer(5, 5, 20, 30);
      convolutionLayer1.set(weights);
      network.add(convolutionLayer1.addRef()).freeRef();
      PoolingLayer poolingLayer1 = new PoolingLayer();
      poolingLayer1.setMode(PoolingLayer.PoolingMode.Max);
      network.add(poolingLayer1.addRef()).freeRef();
      network.add(new ReLuActivationLayer()).freeRef();
      network.add(new ImgCropLayer(18, 18)).freeRef();
      network.add(new NormalizationMetaLayer()).freeRef();

      ConvolutionLayer convolutionLayer = new ConvolutionLayer(3, 3, 30, 40);
      convolutionLayer.set(weights);
      network.add(convolutionLayer.addRef()).freeRef();
      PoolingLayer poolingLayer4 = new PoolingLayer();
      poolingLayer4.setWindowX(4);
      PoolingLayer poolingLayer5 = poolingLayer4.addRef();
      poolingLayer5.setWindowY(4);
      PoolingLayer poolingLayer = poolingLayer5.addRef();
      poolingLayer.setMode(PoolingLayer.PoolingMode.Avg);
      network.add(poolingLayer.addRef()).freeRef();
      network.add(new ReLuActivationLayer()).freeRef();
      network.add(new ImgCropLayer(4, 4)).freeRef();
      network.add(new NormalizationMetaLayer()).freeRef();

      network.add(new ImgBandBiasLayer(40)).freeRef();
      FullyConnectedLayer fullyConnectedLayer = new FullyConnectedLayer(new int[]{4, 4, 40}, new int[]{features});
      fullyConnectedLayer.set(weights);
      network.add(fullyConnectedLayer.addRef()).freeRef();
      network.add(new SoftmaxLayer()).freeRef();

      return network;
    });
  };

  @Nonnull
  public static RevNetworkFactory rev_conv_1 = (log, features) -> {
    log.p("The vector-to-png network uses a fully connected key then a single convolutional key:");
    return log.eval(() -> {
      @Nonnull final PipelineNetwork network = new PipelineNetwork();

      @Nonnull
      IntToDoubleFunction weights = i -> 1e-8 * (Math.random() - 0.5);
      FullyConnectedLayer fullyConnectedLayer = new FullyConnectedLayer(new int[]{features}, new int[]{4, 4, 40});
      fullyConnectedLayer.set(weights);
      network.add(fullyConnectedLayer.addRef()).freeRef();
      network.add(new ImgBandBiasLayer(40)).freeRef();
      network.add(new NormalizationMetaLayer()).freeRef();

      ConvolutionLayer convolutionLayer5 = new ConvolutionLayer(3, 3, 40, 160);
      convolutionLayer5.set(weights);
      network.add(convolutionLayer5.addRef()).freeRef();
      network.add(new ImgReshapeLayer(2, 2, true)).freeRef(); // 8x8x40
      network.add(new ReLuActivationLayer()).freeRef();
      network.add(new NormalizationMetaLayer()).freeRef();

      ConvolutionLayer convolutionLayer4 = new ConvolutionLayer(3, 3, 40, 160);
      convolutionLayer4.set(weights);
      network.add(convolutionLayer4.addRef()).freeRef();
      network.add(new ImgReshapeLayer(2, 2, true)).freeRef(); // 16x16x40
      network.add(new ReLuActivationLayer()).freeRef();
      network.add(new NormalizationMetaLayer()).freeRef();

      ConvolutionLayer convolutionLayer3 = new ConvolutionLayer(3, 3, 40, 160);
      convolutionLayer3.set(weights);
      network.add(convolutionLayer3.addRef()).freeRef();
      network.add(new ImgReshapeLayer(2, 2, true)).freeRef(); // 32x32x40
      network.add(new ReLuActivationLayer()).freeRef();
      network.add(new NormalizationMetaLayer()).freeRef();

      ConvolutionLayer convolutionLayer2 = new ConvolutionLayer(3, 3, 40, 160);
      convolutionLayer2.set(weights);
      network.add(convolutionLayer2.addRef()).freeRef();
      network.add(new ImgReshapeLayer(2, 2, true)).freeRef(); // 64x64x40
      network.add(new ReLuActivationLayer()).freeRef();
      network.add(new NormalizationMetaLayer()).freeRef();

      ConvolutionLayer convolutionLayer1 = new ConvolutionLayer(3, 3, 40, 160);
      convolutionLayer1.set(weights);
      network.add(convolutionLayer1.addRef()).freeRef();
      network.add(new ImgReshapeLayer(2, 2, true)).freeRef(); // 128x128x40
      network.add(new ReLuActivationLayer()).freeRef();
      network.add(new NormalizationMetaLayer()).freeRef();

      ConvolutionLayer convolutionLayer = new ConvolutionLayer(3, 3, 40, 12);
      convolutionLayer.set(weights);
      network.add(convolutionLayer.addRef()).freeRef();
      network.add(new ImgReshapeLayer(2, 2, true)).freeRef(); // 256x256x3
      network.add(new ReLuActivationLayer()).freeRef();

      return network;
    });
  };

  public abstract static class All_Caltech_Tests extends AllTrainingTests {

    public All_Caltech_Tests(final OptimizationStrategy optimizationStrategy, final RevNetworkFactory revFactory,
                             final FwdNetworkFactory fwdFactory) {
      super(fwdFactory, revFactory, optimizationStrategy);
      batchSize = 10;
    }

    @Nonnull
    @Override
    public ImageProblemData getData() {
      return new CaltechProblemData();
    }

    @Nonnull
    @Override
    public CharSequence getDatasetName() {
      return "Caltech101";
    }

    @Nonnull
    @Override
    public ReportType getReportType() {
      return ReportType.Experiments;
    }

    @Nonnull
    @Override
    protected Class<?> getTargetClass() {
      return Caltech101.class;
    }

  }

  public static class QQN extends All_Caltech_Tests {
    public QQN() {
      super(Research.quadratic_quasi_newton, CaltechTests.rev_conv_1, CaltechTests.fwd_conv_1);
    }

    @Nullable
    public static @SuppressWarnings("unused")
    QQN[] addRef(@Nullable QQN[] array) {
      return RefUtil.addRef(array);
    }

    @Override
    protected void intro(@Nonnull final NotebookOutput log) {
      log.p("");
    }
  }

}
