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

package com.simiacryptus.mindseye.opt;

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.MonitoringWrapperLayer;
import com.simiacryptus.mindseye.layers.java.BiasLayer;
import com.simiacryptus.mindseye.layers.java.FullyConnectedLayer;
import com.simiacryptus.mindseye.layers.java.SoftmaxLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.NotebookReportBase;
import com.simiacryptus.mindseye.test.data.MNIST;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.notebook.TableOutput;
import com.simiacryptus.util.JsonUtil;
import com.simiacryptus.util.MonitoredObject;
import com.simiacryptus.util.test.LabeledObject;
import com.simiacryptus.util.test.TestCategories;
import org.junit.Test;
import org.junit.experimental.categories.Category;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import smile.plot.PlotCanvas;
import smile.plot.ScatterPlot;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.io.ByteArrayOutputStream;
import java.io.IOException;

public abstract @com.simiacryptus.ref.lang.RefAware
class MnistTestBase extends NotebookReportBase {
  private static final Logger log = LoggerFactory.getLogger(MnistTestBase.class);

  int modelNo = 0;

  @Nonnull
  @Override
  public ReportType getReportType() {
    return ReportType.Optimizers;
  }

  public static @SuppressWarnings("unused")
  MnistTestBase[] addRefs(MnistTestBase[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(MnistTestBase::addRef)
        .toArray((x) -> new MnistTestBase[x]);
  }

  public static @SuppressWarnings("unused")
  MnistTestBase[][] addRefs(MnistTestBase[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(MnistTestBase::addRefs)
        .toArray((x) -> new MnistTestBase[x][]);
  }

  @Test
  @Category(TestCategories.Report.class)
  public void test() {
    run(this::run);
  }

  public void run(@Nonnull NotebookOutput log) {
    @Nonnull final com.simiacryptus.ref.wrappers.RefList<Step> history = new com.simiacryptus.ref.wrappers.RefArrayList<>();
    @Nonnull final MonitoredObject monitoringRoot = new MonitoredObject();
    @Nonnull final TrainingMonitor monitor = getMonitor(history);
    final Tensor[][] trainingData = getTrainingData(log);
    final DAGNetwork network = buildModel(log);
    addMonitoring(network, monitoringRoot);
    log.h1("Training");
    train(log, network, trainingData, monitor);
    report(log, monitoringRoot, history, network);
    validate(log, network);
    removeMonitoring(network);
  }

  public void addMonitoring(@Nonnull final DAGNetwork network, @Nonnull final MonitoredObject monitoringRoot) {
    network.visitNodes(node -> {
      if (!(node.getLayer() instanceof MonitoringWrapperLayer)) {
        node.setLayer(new MonitoringWrapperLayer(node.getLayer()).addTo(monitoringRoot));
      }
    });
  }

  public DAGNetwork buildModel(@Nonnull final NotebookOutput log) {
    log.h1("Model");
    log.p("This is a very simple model that performs basic logistic regression. "
        + "It is expected to be trainable to about 91% accuracy on MNIST.");
    return log.eval(() -> {
      @Nonnull final PipelineNetwork network = new PipelineNetwork();
      network.add(new BiasLayer(28, 28, 1)).freeRef();
      network.add(
          new FullyConnectedLayer(new int[]{28, 28, 1}, new int[]{10}).set(() -> 0.001 * (Math.random() - 0.45)))
          .freeRef();
      network.add(new SoftmaxLayer()).freeRef();
      return network;
    });
  }

  public Tensor[][] getTrainingData(final NotebookOutput log) {
    Tensor[][] tensors = MNIST.trainingDataStream().map(labeledObject -> {
      @Nonnull final Tensor categoryTensor = new Tensor(10);
      final int category = parse(labeledObject.label);
      categoryTensor.set(category, 1);
      return new Tensor[]{labeledObject.data, categoryTensor};
    }).toArray(i -> new Tensor[i][]);
    return tensors;
  }

  public int parse(@Nonnull final String label) {
    return Integer.parseInt(label.replaceAll("[^\\d]", ""));
  }

  public int[] predict(@Nonnull final Layer network, @Nonnull final LabeledObject<Tensor> labeledObject) {
    @Nullable final double[] predictionSignal = network.eval(labeledObject.data).getData().get(0).getData();
    return com.simiacryptus.ref.wrappers.RefIntStream.range(0, 10).mapToObj(x -> x)
        .sorted(com.simiacryptus.ref.wrappers.RefComparator.comparing(i -> -predictionSignal[i])).mapToInt(x -> x)
        .toArray();
  }

  public void removeMonitoring(@Nonnull final DAGNetwork network) {
    network.visitNodes(node -> {
      if (node.getLayer() instanceof MonitoringWrapperLayer) {
        node.setLayer(((MonitoringWrapperLayer) node.getLayer()).getInner());
      }
    });
  }

  public void report(@Nonnull final NotebookOutput log, @Nonnull final MonitoredObject monitoringRoot,
                     @Nonnull final com.simiacryptus.ref.wrappers.RefList<Step> history, @Nonnull final Layer network) {

    if (!history.isEmpty()) {
      log.eval(() -> {
        @Nonnull final PlotCanvas plot = ScatterPlot
            .plot(history.stream().map(step -> new double[]{step.iteration, Math.log10(step.point.getMean())})
                .toArray(i -> new double[i][]));
        plot.setTitle("Convergence Plot");
        plot.setAxisLabels("Iteration", "log10(Fitness)");
        plot.setSize(600, 400);
        return plot;
      });
    }

    @Nonnull final String modelName = "model" + modelNo++ + ".json";
    log.p("Saved model as " + log.file(network.getJson().toString(), modelName, modelName));

    log.h1("Metrics");
    log.eval(() -> {
      try {
        @Nonnull final ByteArrayOutputStream out = new ByteArrayOutputStream();
        JsonUtil.getMapper().writeValue(out, monitoringRoot.getMetrics());
        return out.toString();
      } catch (@Nonnull final IOException e) {
        throw new RuntimeException(e);
      }
    });
  }

  @Nonnull
  public TrainingMonitor getMonitor(@Nonnull final com.simiacryptus.ref.wrappers.RefList<Step> history) {
    return new TrainingMonitor() {
      @Override
      public void clear() {
        super.clear();
      }

      @Override
      public void log(final String msg) {
        log.info(msg);
        super.log(msg);
      }

      @Override
      public void onStepComplete(final Step currentPoint) {
        history.add(currentPoint);
        super.onStepComplete(currentPoint);
      }
    };
  }

  public abstract void train(NotebookOutput log, Layer network, Tensor[][] trainingData, TrainingMonitor monitor);

  public void validate(@Nonnull final NotebookOutput log, @Nonnull final Layer network) {
    log.h1("Validation");
    log.p("If we apply our model against the entire validation dataset, we get this accuracy:");
    log.eval(() -> {
      return MNIST.validationDataStream()
          .mapToDouble(labeledObject -> predict(network, labeledObject)[0] == parse(labeledObject.label) ? 1 : 0)
          .average().getAsDouble() * 100;
    });

    log.p("Let's examine some incorrectly predicted results in more detail:");
    log.eval(() -> {
      @Nonnull final TableOutput table = new TableOutput();
      MNIST.validationDataStream().map(labeledObject -> {
        final int actualCategory = parse(labeledObject.label);
        @Nullable final double[] predictionSignal = network.eval(labeledObject.data).getData().get(0).getData();
        final int[] predictionList = com.simiacryptus.ref.wrappers.RefIntStream.range(0, 10).mapToObj(x -> x)
            .sorted(com.simiacryptus.ref.wrappers.RefComparator.comparing(i -> -predictionSignal[i])).mapToInt(x -> x)
            .toArray();
        if (predictionList[0] == actualCategory)
          return null; // We will only examine mispredicted rows
        @Nonnull final com.simiacryptus.ref.wrappers.RefLinkedHashMap<CharSequence, Object> row = new com.simiacryptus.ref.wrappers.RefLinkedHashMap<>();
        row.put("Image", log.png(labeledObject.data.toGrayImage(), labeledObject.label));
        row.put("Prediction",
            com.simiacryptus.ref.wrappers.RefArrays.stream(predictionList).limit(3)
                .mapToObj(i -> String.format("%d (%.1f%%)", i, 100.0 * predictionSignal[i]))
                .reduce((a, b) -> a + ", " + b).get());
        return row;
      }).filter(x -> null != x).limit(10).forEach(table::putRow);
      return table;
    });
  }

  public @SuppressWarnings("unused")
  void _free() {
  }

  public @Override
  @SuppressWarnings("unused")
  MnistTestBase addRef() {
    return (MnistTestBase) super.addRef();
  }
}
