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
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.mindseye.layers.MonitoringWrapperLayer;
import com.simiacryptus.mindseye.layers.java.BiasLayer;
import com.simiacryptus.mindseye.layers.java.FullyConnectedLayer;
import com.simiacryptus.mindseye.layers.java.SoftmaxLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.data.MNIST;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.notebook.TableOutput;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.*;
import com.simiacryptus.util.JsonUtil;
import com.simiacryptus.util.MonitoredObject;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.test.LabeledObject;
import com.simiacryptus.util.test.NotebookTestBase;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import smile.plot.swing.Canvas;
import smile.plot.swing.PlotPanel;
import smile.plot.swing.ScatterPlot;

import javax.annotation.Nonnull;
import java.io.ByteArrayOutputStream;
import java.io.IOException;

public abstract class MnistTestBase extends NotebookTestBase {
  private static final Logger log = LoggerFactory.getLogger(MnistTestBase.class);

  int modelNo = 0;

  @Nonnull
  @Override
  public ReportType getReportType() {
    return ReportType.Optimizers;
  }

  @Nonnull
  public Tensor[][] getTrainingData() {
    return MNIST.trainingDataStream().map(labeledObject -> {
      @Nonnull final Tensor categoryTensor = new Tensor(10);
      final int category = parse(labeledObject.label);
      categoryTensor.set(category, 1);
      return new Tensor[]{labeledObject.data, categoryTensor};
    }).toArray(i -> new Tensor[i][]);
  }

  @Test
  @Tag("Report")
  public void test() {
    @Nonnull NotebookOutput log = getLog();
    @Nonnull final RefList<Step> history = new RefArrayList<>();
    @Nonnull final MonitoredObject monitoringRoot = new MonitoredObject();
    @Nonnull final TrainingMonitor monitor = getMonitor(history);
    final Tensor[][] trainingData = getTrainingData();
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
        node.setLayer(new MonitoringWrapperLayer(node.getLayer()).addTo2(monitoringRoot));
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
      FullyConnectedLayer fullyConnectedLayer = new FullyConnectedLayer(new int[]{28, 28, 1}, new int[]{10});
      fullyConnectedLayer.set(() -> 0.001 * (Math.random() - 0.45));
      network.add(
          fullyConnectedLayer.addRef())
          .freeRef();
      network.add(new SoftmaxLayer()).freeRef();
      return network;
    });
  }

  public int parse(@Nonnull final String label) {
    return Integer.parseInt(label.replaceAll("[^\\d]", ""));
  }

  public int[] predict(@Nonnull final Layer network, @Nonnull final LabeledObject<Tensor> labeledObject) {
    Result eval = network.eval(labeledObject.data);
    TensorList data = eval.getData();
    Tensor tensor = data.get(0);
    data.freeRef();
    eval.freeRef();
    return RefIntStream.range(0, 10)
        .mapToObj(x -> x)
        .sorted(RefComparator.comparingDouble(i -> -tensor.get(i)))
        .mapToInt(x -> x)
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
                     @Nonnull final RefList<Step> history, @Nonnull final Layer network) {

    if (!history.isEmpty()) {
      log.eval(() -> {
        @Nonnull final ScatterPlot plot = ScatterPlot
            .of(history.stream().map(step -> {
              assert step.point != null;
              return new double[]{step.iteration, Math.log10(step.point.getMean())};
            })
                .toArray(i -> new double[i][]));
        Canvas canvas = new Canvas(new double[]{0,0}, new double[]{1,1});
        canvas.add(plot);
        PlotPanel plotPanel = new PlotPanel(canvas);
        canvas.setTitle("Convergence Plot");
        canvas.setAxisLabels("Iteration", "log10(Fitness)");
        plotPanel.setSize(600, 400);
        return plotPanel;
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
        throw Util.throwException(e);
      }
    });
  }

  @Nonnull
  public TrainingMonitor getMonitor(@Nonnull final RefList<Step> history) {
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
        Result eval = network.eval(labeledObject.data);
        TensorList data = eval.getData();
        Tensor tensor = data.get(0);
        data.freeRef();
        eval.freeRef();
        final int[] predictionList = RefIntStream.range(0, 10).mapToObj(x -> x)
            .sorted(RefComparator.comparingDouble(i -> -tensor.get(i)))
            .mapToInt(x -> x).toArray();
        if (predictionList[0] == actualCategory)
          return null; // We will only examine mispredicted rows
        @Nonnull final RefLinkedHashMap<CharSequence, Object> row = new RefLinkedHashMap<>();
        row.put("Image", log.png(labeledObject.data.toGrayImage(), labeledObject.label));
        row.put("Prediction",
            RefUtil.get(RefArrays.stream(predictionList).limit(3)
                .mapToObj(i -> RefString.format("%d (%.1f%%)", i, 100.0 * tensor.get(i)))
                .reduce((a, b) -> a + ", " + b)));
        return row;
      }).filter(x -> null != x).limit(10).forEach(properties -> table.putRow(properties));
      return table;
    });
  }

}
