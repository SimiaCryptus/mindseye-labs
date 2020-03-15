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

package com.simiacryptus.mindseye.test.integration;

import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.SampledArrayTrainable;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.layers.java.*;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.ValidatingTrainer;
import com.simiacryptus.mindseye.test.StepRecord;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.util.ImageUtil;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.notebook.TableOutput;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.data.ScalarStatistics;
import guru.nidi.graphviz.engine.Format;
import guru.nidi.graphviz.engine.Graphviz;
import guru.nidi.graphviz.model.Graph;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import javax.imageio.ImageIO;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class EncodingProblem implements Problem {

  private static int modelNo = 0;
  private final ImageProblemData data;
  private final List<StepRecord> history = new ArrayList<>();
  private final OptimizationStrategy optimizer;
  private final RevNetworkFactory revFactory;
  private int batchSize = 10000;
  private int features;
  private int timeoutMinutes = 1;
  private int trainingSize = 15000;

  public EncodingProblem(final RevNetworkFactory revFactory, final OptimizationStrategy optimizer,
                         final ImageProblemData data, final int features) {
    this.revFactory = revFactory;
    this.optimizer = optimizer;
    this.data = data;
    this.features = features;
  }

  public int getBatchSize() {
    return batchSize;
  }

  @Nonnull
  public EncodingProblem setBatchSize(final int batchSize) {
    this.batchSize = batchSize;
    return this;
  }

  public int getFeatures() {
    return features;
  }

  @Nonnull
  public EncodingProblem setFeatures(final int features) {
    this.features = features;
    return this;
  }

  @Nonnull
  @Override
  public List<StepRecord> getHistory() {
    return history;
  }

  public int getTimeoutMinutes() {
    return timeoutMinutes;
  }

  @Nonnull
  public EncodingProblem setTimeoutMinutes(final int timeoutMinutes) {
    this.timeoutMinutes = timeoutMinutes;
    return this;
  }

  public int getTrainingSize() {
    return trainingSize;
  }

  @Nonnull
  public EncodingProblem setTrainingSize(final int trainingSize) {
    this.trainingSize = trainingSize;
    return this;
  }

  public double random() {
    return 0.1 * (Math.random() - 0.5);
  }

  @Nonnull
  @Override
  public EncodingProblem run(@Nonnull final NotebookOutput log) {
    @Nonnull final TrainingMonitor monitor = TestUtil.getMonitor(history);
    Tensor[][] trainingData;
    try {
      trainingData = data.trainingData().map(labeledObject -> {
        Tensor tensor = new Tensor(features);
        tensor.set(() -> random());
        return new Tensor[]{tensor.addRef(), labeledObject.data};
      }).toArray(i -> new Tensor[i][]);
    } catch (@Nonnull final IOException e) {
      throw new RuntimeException(e);
    }

    @Nonnull final DAGNetwork imageNetwork = revFactory.vectorToImage(log, features);
    log.h3("Network Diagram");
    log.eval(() -> {
      return Graphviz.fromGraph((Graph) TestUtil.toGraph(imageNetwork)).height(400).width(600).render(Format.PNG)
          .toImage();
    });

    @Nonnull final PipelineNetwork trainingNetwork = new PipelineNetwork(2);
    @Nullable final DAGNode image = trainingNetwork.add(imageNetwork, trainingNetwork.getInput(0));
    @Nullable final DAGNode softmax = trainingNetwork.add(new SoftmaxLayer(), trainingNetwork.getInput(0));
    NthPowerActivationLayer nthPowerActivationLayer = new NthPowerActivationLayer();
    nthPowerActivationLayer.setPower(1.0 / 2.0);
    trainingNetwork.add(new SumInputsLayer(), trainingNetwork.add(new EntropyLossLayer(), softmax, softmax),
        trainingNetwork.add(nthPowerActivationLayer.addRef(),
            trainingNetwork.add(new MeanSqLossLayer(), image, trainingNetwork.getInput(1))))
        .freeRef();
    log.h3("Training");
    log.p("We start by training apply a very small population to improve initial convergence performance:");
    TestUtil.instrumentPerformance(trainingNetwork);
    @Nonnull final Tensor[][] primingData = RefArrays.copyOfRange(trainingData, 0, 1000);
    SampledArrayTrainable sampledArrayTrainable1 = new SampledArrayTrainable(primingData, trainingNetwork, trainingSize, batchSize);
    sampledArrayTrainable1.setMinSamples(trainingSize);
    sampledArrayTrainable1.setMask(true, false);
    @Nonnull final ValidatingTrainer preTrainer = optimizer.train(log,
        sampledArrayTrainable1.addRef(),
        new ArrayTrainable(primingData, trainingNetwork, batchSize), monitor);
    log.run(() -> {
      preTrainer.setTimeout(timeoutMinutes / 2, TimeUnit.MINUTES);
      ValidatingTrainer validatingTrainer = preTrainer.addRef();
      validatingTrainer.setMaxIterations(batchSize);
      validatingTrainer.addRef().run();
    });
    TestUtil.extractPerformance(log, trainingNetwork);

    log.p("Then our main training phase:");
    TestUtil.instrumentPerformance(trainingNetwork);
    SampledArrayTrainable sampledArrayTrainable = new SampledArrayTrainable(trainingData, trainingNetwork, trainingSize, batchSize);
    sampledArrayTrainable.setMinSamples(trainingSize);
    sampledArrayTrainable.setMask(true, false);
    @Nonnull final ValidatingTrainer mainTrainer = optimizer.train(log,
        sampledArrayTrainable,
        new ArrayTrainable(trainingData, trainingNetwork, batchSize), monitor);
    log.run(() -> {
      mainTrainer.setTimeout(timeoutMinutes, TimeUnit.MINUTES);
      ValidatingTrainer validatingTrainer = mainTrainer.addRef();
      validatingTrainer.setMaxIterations(batchSize);
      validatingTrainer.addRef().run();
    });
    TestUtil.extractPerformance(log, trainingNetwork);

    if (!history.isEmpty()) {
      log.eval(() -> {
        return TestUtil.plot(history);
      });
      log.eval(() -> {
        return TestUtil.plotTime(history);
      });
    }

    try {
      @Nonnull
      String filename = log.getFileName() + EncodingProblem.modelNo++ + "_plot.png";
      ImageIO.write(Util.toImage(TestUtil.plot(history)), "png", log.file(filename));
      log.appendMetadata("result_plot", filename, ";");
    } catch (IOException e) {
      throw new RuntimeException(e);
    }

    //log.file()
    @Nonnull final String modelName = "encoding_model_" + EncodingProblem.modelNo++ + ".json";
    log.appendMetadata("result_model", modelName, ";");
    log.p("Saved model as " + log.file(trainingNetwork.getJson().toString(), modelName, modelName));

    log.h3("Results");
    @Nonnull final PipelineNetwork testNetwork = new PipelineNetwork(2);
    testNetwork.add(imageNetwork, testNetwork.getInput(0));
    log.eval(() -> {
      @Nonnull final TableOutput table = new TableOutput();
      RefArrays.stream(trainingData).map(tensorArray -> {
        @Nullable final Tensor predictionSignal = testNetwork.eval(tensorArray).getData().get(0);
        @Nonnull final LinkedHashMap<CharSequence, Object> row = new LinkedHashMap<>();
        row.put("Source", log.png(tensorArray[1].toImage(), ""));
        row.put("Echo", log.png(predictionSignal.toImage(), ""));
        return row;
      }).filter(x -> true).limit(10).forEach(properties -> table.putRow(properties));
      return table;
    });

    log.p("Learned Model Statistics:");
    log.eval(() -> {
      @Nonnull final ScalarStatistics scalarStatistics = new ScalarStatistics();
      trainingNetwork.state().stream().flatMapToDouble(x -> Arrays.stream(x)).forEach(v -> scalarStatistics.add(v));
      return scalarStatistics.getMetrics();
    });

    log.p("Learned Representation Statistics:");
    log.eval(() -> {
      @Nonnull final ScalarStatistics scalarStatistics = new ScalarStatistics();
      RefArrays.stream(trainingData).flatMapToDouble(row -> row[0].doubleStream())
          .forEach(v -> scalarStatistics.add(v));
      return scalarStatistics.getMetrics();
    });

    log.p("Some rendered unit vectors:");
    for (int featureNumber = 0; featureNumber < features; featureNumber++) {
      Tensor tensor1 = new Tensor(features);
      tensor1.set(featureNumber, 1);
      @Nonnull final Tensor input = tensor1.addRef();
      @Nullable final Tensor tensor = imageNetwork.eval(input).getData().get(0);
      ImageUtil.renderToImages(tensor, true).forEach(img -> {
        log.out(log.png(img, ""));
      });
    }

    return this;
  }
}
