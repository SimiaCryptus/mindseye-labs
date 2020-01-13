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
import com.simiacryptus.ref.lang.RefAware;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.function.IntToDoubleFunction;

public class CaltechTests {

  @Nonnull
  public static FwdNetworkFactory fwd_conv_1 = (log, features) -> {
    log.p("The png-to-vector network is a single key convolutional:");
    return log.eval(() -> {
      @Nonnull
      final PipelineNetwork network = new PipelineNetwork();

      @Nonnull
      IntToDoubleFunction weights = i -> 1e-8 * (Math.random() - 0.5);
      network.add(new ConvolutionLayer(3, 3, 3, 10).set(weights)).freeRef();
      network.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max)).freeRef();
      network.add(new ReLuActivationLayer()).freeRef();
      network.add(new ImgCropLayer(126, 126)).freeRef();
      network.add(new NormalizationMetaLayer()).freeRef();

      network.add(new ConvolutionLayer(3, 3, 10, 20).set(weights)).freeRef();
      network.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max)).freeRef();
      network.add(new ReLuActivationLayer()).freeRef();
      network.add(new ImgCropLayer(62, 62)).freeRef();
      network.add(new NormalizationMetaLayer()).freeRef();

      network.add(new ConvolutionLayer(5, 5, 20, 30).set(weights)).freeRef();
      network.add(new PoolingLayer().setMode(PoolingLayer.PoolingMode.Max)).freeRef();
      network.add(new ReLuActivationLayer()).freeRef();
      network.add(new ImgCropLayer(18, 18)).freeRef();
      network.add(new NormalizationMetaLayer()).freeRef();

      network.add(new ConvolutionLayer(3, 3, 30, 40).set(weights)).freeRef();
      network.add(new PoolingLayer().setWindowX(4).setWindowY(4).setMode(PoolingLayer.PoolingMode.Avg)).freeRef();
      network.add(new ReLuActivationLayer()).freeRef();
      network.add(new ImgCropLayer(4, 4)).freeRef();
      network.add(new NormalizationMetaLayer()).freeRef();

      network.add(new ImgBandBiasLayer(40)).freeRef();
      network.add(new FullyConnectedLayer(new int[] { 4, 4, 40 }, new int[] { features }).set(weights)).freeRef();
      network.add(new SoftmaxLayer()).freeRef();

      return network;
    });
  };

  @Nonnull
  public static RevNetworkFactory rev_conv_1 = (log, features) -> {
    log.p("The vector-to-png network uses a fully connected key then a single convolutional key:");
    return log.eval(() -> {
      @Nonnull
      final PipelineNetwork network = new PipelineNetwork();

      @Nonnull
      IntToDoubleFunction weights = i -> 1e-8 * (Math.random() - 0.5);
      network.add(new FullyConnectedLayer(new int[] { features }, new int[] { 4, 4, 40 }).set(weights)).freeRef();
      network.add(new ImgBandBiasLayer(40)).freeRef();
      network.add(new NormalizationMetaLayer()).freeRef();

      network.add(new ConvolutionLayer(3, 3, 40, 160).set(weights)).freeRef();
      network.add(new ImgReshapeLayer(2, 2, true)).freeRef(); // 8x8x40
      network.add(new ReLuActivationLayer()).freeRef();
      network.add(new NormalizationMetaLayer()).freeRef();

      network.add(new ConvolutionLayer(3, 3, 40, 160).set(weights)).freeRef();
      network.add(new ImgReshapeLayer(2, 2, true)).freeRef(); // 16x16x40
      network.add(new ReLuActivationLayer()).freeRef();
      network.add(new NormalizationMetaLayer()).freeRef();

      network.add(new ConvolutionLayer(3, 3, 40, 160).set(weights)).freeRef();
      network.add(new ImgReshapeLayer(2, 2, true)).freeRef(); // 32x32x40
      network.add(new ReLuActivationLayer()).freeRef();
      network.add(new NormalizationMetaLayer()).freeRef();

      network.add(new ConvolutionLayer(3, 3, 40, 160).set(weights)).freeRef();
      network.add(new ImgReshapeLayer(2, 2, true)).freeRef(); // 64x64x40
      network.add(new ReLuActivationLayer()).freeRef();
      network.add(new NormalizationMetaLayer()).freeRef();

      network.add(new ConvolutionLayer(3, 3, 40, 160).set(weights)).freeRef();
      network.add(new ImgReshapeLayer(2, 2, true)).freeRef(); // 128x128x40
      network.add(new ReLuActivationLayer()).freeRef();
      network.add(new NormalizationMetaLayer()).freeRef();

      network.add(new ConvolutionLayer(3, 3, 40, 12).set(weights)).freeRef();
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

    public static @SuppressWarnings("unused") All_Caltech_Tests[] addRefs(All_Caltech_Tests[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(All_Caltech_Tests::addRef)
          .toArray((x) -> new All_Caltech_Tests[x]);
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") All_Caltech_Tests addRef() {
      return (All_Caltech_Tests) super.addRef();
    }

  }

  public static class QQN extends All_Caltech_Tests {
    public QQN() {
      super(Research.quadratic_quasi_newton, CaltechTests.rev_conv_1, CaltechTests.fwd_conv_1);
    }

    public static @SuppressWarnings("unused") QQN[] addRefs(QQN[] array) {
      if (array == null)
        return null;
      return Arrays.stream(array).filter((x) -> x != null).map(QQN::addRef).toArray((x) -> new QQN[x]);
    }

    public @SuppressWarnings("unused") void _free() {
    }

    public @Override @SuppressWarnings("unused") QQN addRef() {
      return (QQN) super.addRef();
    }

    @Override
    protected void intro(@Nonnull final NotebookOutput log) {
      log.p("");
    }

  }

}
