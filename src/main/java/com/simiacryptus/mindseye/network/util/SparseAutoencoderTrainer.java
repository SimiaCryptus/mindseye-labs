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

package com.simiacryptus.mindseye.network.util;

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.layers.java.BinaryNoiseLayer;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.layers.java.MeanSqLossLayer;
import com.simiacryptus.mindseye.layers.java.SumReducerLayer;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.SupervisedNetwork;

import javax.annotation.Nonnull;

@SuppressWarnings("serial")
public class SparseAutoencoderTrainer extends SupervisedNetwork {

  public final DAGNode decoder;
  public final DAGNode encoder;
  public final DAGNode loss;
  public final DAGNode sparsity;
  public final DAGNode sparsityThrottleLayer;
  public final DAGNode sumFitnessLayer;
  public final DAGNode sumSparsityLayer;

  public SparseAutoencoderTrainer(@Nonnull final Layer encoder, @Nonnull final Layer decoder) {
    super(1);
    this.encoder = add(encoder, getInput(0));
    this.decoder = add(decoder, this.encoder);
    loss = add(new MeanSqLossLayer(), this.decoder, getInput(0));
    sparsity = add(new BinaryNoiseLayer(), this.encoder);
    sumSparsityLayer = add(new SumReducerLayer(), sparsity);
    sparsityThrottleLayer = add(new LinearActivationLayer().setScale(0.5), sumSparsityLayer);
    sumFitnessLayer = add(new SumReducerLayer(), sparsityThrottleLayer, loss);
  }

  @Override
  public DAGNode getHead() {
    sumFitnessLayer.addRef();
    return sumFitnessLayer;
  }
}
