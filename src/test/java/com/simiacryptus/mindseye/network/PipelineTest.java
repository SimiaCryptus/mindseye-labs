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
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.java.MeanSqLossLayer;
import com.simiacryptus.mindseye.test.NotebookReportBase;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.test.unit.SerializationTest;
import com.simiacryptus.mindseye.test.unit.TrainingTester;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.ref.wrappers.RefArrayList;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefList;
import com.simiacryptus.ref.wrappers.RefString;
import com.simiacryptus.util.Util;
import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;
import guru.nidi.graphviz.model.Graph;
import org.junit.jupiter.api.Test;

import javax.annotation.Nonnull;

public abstract class PipelineTest extends NotebookReportBase {

  final RefList<Layer> pipeline;

  public PipelineTest(final RefList<Layer> pipeline) {
    this.pipeline = pipeline;
  }

  public PipelineTest(final Layer... pipeline) {
    this(RefArrays.asList(pipeline));
  }

  @Nonnull
  public abstract int[] getInputDims();

  @Nonnull
  public Layer buildNetwork(@Nonnull final Layer... layers) {
    @Nonnull final PipelineNetwork network = new PipelineNetwork(1);
    for (@Nonnull final Layer layer : layers) {
      network.add(layer.copy());
    }
    return network;
  }

  public void graphviz(@Nonnull final NotebookOutput log, final Layer layer) {
    if (layer instanceof DAGNetwork) {
      log.p("This is a network apply the following layout:");
      log.eval(() -> {
        return Graphviz.fromGraph((Graph) TestUtil.toGraph((DAGNetwork) layer)).height(400).width(600)
            .render(Format.PNG).toImage();
      });
    }
  }

  public double random() {
    return Math.round(1000.0 * (Util.R.get().nextDouble() - 0.5)) / 250.0;
  }

  @Nonnull
  public Tensor[] randomize(@Nonnull final int[][] inputDims) {
    return RefArrays.stream(inputDims).map(dim -> {
      Tensor tensor = new Tensor(dim);
      tensor.set(() -> random());
      return tensor.addRef();
    }).toArray(i -> new Tensor[i]);
  }

  @Test
  public void test() {
    @Nonnull final NotebookOutput log = getLog();
    @Nonnull final RefArrayList<Layer> workingSpec = new RefArrayList<>();
    int layerIndex = 0;
    for (final Layer l : pipeline) {
      workingSpec.add(l);
      @Nonnull final Layer networkHead = buildNetwork(workingSpec.toArray(new Layer[]{}));
      graphviz(log, networkHead);
      test(log, networkHead, RefString.format("Pipeline Network apply %d Layers", layerIndex++), getInputDims());
    }
  }

  public TrainingTester.ComponentResult test(@Nonnull final NotebookOutput log, @Nonnull final Layer layer,
                                             final String header, @Nonnull final int[]... inputDims) {
    @Nonnull final Layer component = layer.copy();
    final Tensor[] randomize = randomize(inputDims);
    new SerializationTest().test(log, component, randomize);
    return new TrainingTester() {
      public @SuppressWarnings("unused")
      void _free() {
      }

      @Override
      protected void printHeader(@Nonnull NotebookOutput log) {
        log.h1(header);
      }

      @Nonnull
      @Override
      protected Layer lossLayer() {
        return new MeanSqLossLayer();
      }
    }.test(log, component, randomize);
  }

}
