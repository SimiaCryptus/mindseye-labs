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
import com.simiacryptus.notebook.MarkdownNotebookOutput;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.test.SysOutInterceptor;
import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;
import guru.nidi.graphviz.model.Graph;
import org.junit.Test;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

public abstract @com.simiacryptus.ref.lang.RefAware
class NLayerTest {
  static {
    SysOutInterceptor.INSTANCE.init();
  }

  protected final String reportingFolder = "reports/_reports";
  @Nonnull
  final List<int[]> dimList;

  public NLayerTest(final int[]... dimList) {
    this.dimList = Arrays.asList(dimList);
  }

  @Nonnull
  public abstract int[] getInputDims();

  public abstract void addLayer(PipelineNetwork network, int[] in, int[] out);

  @Nonnull
  public Layer buildNetwork(@Nonnull final int[]... dimList) {
    @Nonnull final PipelineNetwork network = new PipelineNetwork(1);
    @Nullable
    int[] last = null;
    for (final int[] dims : dimList) {
      if (null != last) {
        addLayer(network, last, dims);
      }
      last = dims;
    }
    return network;
  }

  public int[][] concat(final int[] a, @Nonnull final List<int[]> b) {
    return Stream.concat(Stream.of(a), b.stream())
        .toArray(i -> new int[i][]);
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

  public Tensor[] randomize(@Nonnull final int[][] inputDims) {
    return com.simiacryptus.ref.wrappers.RefArrays.stream(inputDims).map(dim -> new Tensor(dim).set(this::random))
        .toArray(i -> new Tensor[i]);
  }

  @Test
  public void test() throws Throwable {
    try (@Nonnull
         NotebookOutput log = MarkdownNotebookOutput
        .get(NotebookReportBase.getTestReportLocation(((Object) this).getClass(), reportingFolder))) {
      test(log);
    }
  }

  public void test(@Nonnull final NotebookOutput log) {

    log.h1("%s", getClass().getSimpleName());
    @Nonnull final int[] inputDims = getInputDims();
    @Nonnull final ArrayList<int[]> workingSpec = new ArrayList<>();
    for (final int[] l : dimList) {
      workingSpec.add(l);
      @Nonnull final Layer layer = buildNetwork(concat(inputDims, workingSpec));
      graphviz(log, layer);
      test(log, layer, inputDims);
    }
  }

  @Nullable
  public TrainingTester.ComponentResult test(@Nonnull final NotebookOutput log, @Nonnull final Layer layer,
                                             @Nonnull final int[]... inputDims) {
    @Nonnull final Layer component = layer.copy();
    final Tensor[] randomize = randomize(inputDims);
    new SerializationTest().test(log, component, randomize);
    return new TrainingTester() {
      public @SuppressWarnings("unused")
      void _free() {
      }

      @Override
      protected Layer lossLayer() {
        return new MeanSqLossLayer();
      }
    }.test(log, component, randomize);
  }

}
