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

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.layers.java.*;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.util.FastRandom;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.UUID;

@SuppressWarnings("serial")
public @com.simiacryptus.ref.lang.RefAware
class SigmoidTreeNetwork extends DAGNetwork implements EvolvingNetwork {

  private final boolean multigate = false;
  double initialFuzzyCoeff = 1e-8;
  @Nullable
  private Layer alpha = null;
  @Nullable
  private Layer alphaBias = null;
  @Nullable
  private Layer beta = null;
  @Nullable
  private Layer betaBias = null;
  @Nullable
  private Layer gate = null;
  @Nullable
  private Layer gateBias = null;
  @Nullable
  private DAGNode head = null;
  @Nullable
  private NodeMode mode = null;
  private boolean skipChildStage = true;
  private boolean skipFuzzy = false;

  protected SigmoidTreeNetwork(@Nonnull final JsonObject json,
                               com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    super(json, rs);
    head = getNodeById(UUID.fromString(json.get("head").getAsString()));
    com.simiacryptus.ref.wrappers.RefMap<UUID, Layer> layersById = getLayersById();
    if (json.get("alpha") != null) {
      alpha = layersById.get(UUID.fromString(json.get("alpha").getAsString()));
    }
    if (json.get("alphaBias") != null) {
      alphaBias = layersById.get(UUID.fromString(json.get("alphaBias").getAsString()));
    }
    if (json.get("beta") != null) {
      beta = layersById.get(UUID.fromString(json.get("beta").getAsString()));
    }
    if (json.get("betaBias") != null) {
      betaBias = layersById.get(UUID.fromString(json.get("betaBias").getAsString()));
    }
    if (json.get("gate") != null) {
      gate = layersById.get(UUID.fromString(json.get("gate").getAsString()));
    }
    if (json.get("gateBias") != null) {
      gate = layersById.get(UUID.fromString(json.get("gateBias").getAsString()));
    }
    setSkipChildStage(
        json.get("skipChildStage") != null ? json.get("skipChildStage").getAsBoolean() : skipChildStage());
    setSkipFuzzy(json.get("skipFuzzy") != null ? json.get("skipFuzzy").getAsBoolean() : isSkipFuzzy());
    mode = NodeMode.valueOf(json.get("mode").getAsString());
  }

  public SigmoidTreeNetwork(final Layer alpha, final Layer alphaBias) {
    super(1);
    this.alpha = alpha;
    this.alphaBias = alphaBias;
    mode = NodeMode.Linear;
  }

  @Nullable
  @Override
  public synchronized DAGNode getHead() {
    if (null == head) {
      synchronized (this) {
        if (null == head) {
          reset();
          final DAGNode input = getInput(0);
          switch (getMode()) {
            case Linear:
              head = add(alpha.setFrozen(false), add(alphaBias.setFrozen(false), input));
              break;
            case Fuzzy: {
              final DAGNode gateNode = add(gate.setFrozen(false),
                  null != gateBias ? add(gateBias.setFrozen(false), input) : input);
              head = add(new ProductInputsLayer(), add(alpha.setFrozen(false), add(alphaBias.setFrozen(false), input)),
                  add(new LinearActivationLayer().setScale(2).freeze(),
                      add(new SigmoidActivationLayer().setBalanced(false), gateNode)));
              break;
            }
            case Bilinear: {
              final DAGNode gateNode = add(gate.setFrozen(false),
                  null != gateBias ? add(gateBias.setFrozen(false), input) : input);
              head = add(new SumInputsLayer(),
                  add(new ProductInputsLayer(), add(alpha.setFrozen(false), add(alphaBias.setFrozen(false), input)),
                      add(new SigmoidActivationLayer().setBalanced(false), gateNode)),
                  add(new ProductInputsLayer(), add(beta.setFrozen(false), add(betaBias.setFrozen(false), input)),
                      add(new SigmoidActivationLayer().setBalanced(false),
                          add(new LinearActivationLayer().setScale(-1).freeze(), gateNode))));
              break;
            }
            case Final:
              final DAGNode gateNode = add(gate.setFrozen(false),
                  null != gateBias ? add(gateBias.setFrozen(false), input) : input);
              head = add(new SumInputsLayer(),
                  add(new ProductInputsLayer(), add(alpha, input),
                      add(new SigmoidActivationLayer().setBalanced(false), gateNode)),
                  add(new ProductInputsLayer(), add(beta, input), add(new SigmoidActivationLayer().setBalanced(false),
                      add(new LinearActivationLayer().setScale(-1).freeze(), gateNode))));
              break;
          }
        }
      }
    }
    head.addRef();
    return head;
  }

  @Nullable
  public NodeMode getMode() {
    return mode;
  }

  public boolean isSkipFuzzy() {
    return skipFuzzy;
  }

  @Nonnull
  public SigmoidTreeNetwork setSkipFuzzy(final boolean skipFuzzy) {
    this.skipFuzzy = skipFuzzy;
    return this;
  }

  @Nonnull
  public SigmoidTreeNetwork setSkipChildStage(final boolean skipChildStage) {
    this.skipChildStage = skipChildStage;
    return this;
  }

  public static SigmoidTreeNetwork fromJson(@Nonnull final JsonObject json,
                                            com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> rs) {
    return new SigmoidTreeNetwork(json, rs);
  }

  public static @SuppressWarnings("unused")
  SigmoidTreeNetwork[] addRefs(SigmoidTreeNetwork[] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(SigmoidTreeNetwork::addRef)
        .toArray((x) -> new SigmoidTreeNetwork[x]);
  }

  public static @SuppressWarnings("unused")
  SigmoidTreeNetwork[][] addRefs(SigmoidTreeNetwork[][] array) {
    if (array == null)
      return null;
    return java.util.Arrays.stream(array).filter((x) -> x != null).map(SigmoidTreeNetwork::addRefs)
        .toArray((x) -> new SigmoidTreeNetwork[x][]);
  }

  public void _free() {
    head.freeRef();
    alpha.freeRef();
    alphaBias.freeRef();
    beta.freeRef();
    betaBias.freeRef();
    gate.freeRef();
    super._free();
  }

  public void copyState(@Nonnull final Layer from, @Nonnull final Layer to) {
    @Nullable final com.simiacryptus.ref.wrappers.RefList<double[]> alphaState = from.state();
    @Nullable final com.simiacryptus.ref.wrappers.RefList<double[]> betaState = to.state();
    for (int i = 0; i < alphaState.size(); i++) {
      final double[] betaBuffer = betaState.get(i);
      final double[] alphaBuffer = alphaState.get(i);
      System.arraycopy(alphaBuffer, 0, betaBuffer, 0, alphaBuffer.length);
    }
  }

  @Override
  public JsonObject getJson(com.simiacryptus.ref.wrappers.RefMap<CharSequence, byte[]> resources,
                            DataSerializer dataSerializer) {
    assertConsistent();
    UUID headId = getHeadId();
    final JsonObject json = super.getJson(resources, dataSerializer);
    json.addProperty("head", headId.toString());
    if (null != alpha) {
      json.addProperty("alpha", alpha.getId().toString());
    }
    if (null != alphaBias) {
      json.addProperty("alphaBias", alpha.getId().toString());
    }
    if (null != beta) {
      json.addProperty("beta", beta.getId().toString());
    }
    if (null != betaBias) {
      json.addProperty("betaBias", beta.getId().toString());
    }
    if (null != gate) {
      json.addProperty("gate", gate.getId().toString());
    }
    if (null != gateBias) {
      json.addProperty("gateBias", gate.getId().toString());
    }
    json.addProperty("mode", getMode().name());
    json.addProperty("skipChildStage", skipChildStage());
    json.addProperty("skipFuzzy", isSkipFuzzy());
    assert null != Layer.fromJson(json) : "Smoke apply deserialization";
    return json;
  }

  @Override
  public void nextPhase() {
    switch (getMode()) {
      case Linear: {
        head = null;
        @Nonnull final FullyConnectedLayer alpha = (FullyConnectedLayer) this.alpha;
        //alphaList.weights.scale(2);
        gate = new FullyConnectedLayer(alpha.inputDims, multigate ? alpha.outputDims : new int[]{1});
        gateBias = new BiasLayer(alpha.inputDims);
        mode = NodeMode.Fuzzy;
        break;
      }
      case Fuzzy: {
        head = null;
        @Nullable final FullyConnectedLayer alpha = (FullyConnectedLayer) this.alpha;
        @Nonnull final BiasLayer alphaBias = (BiasLayer) this.alphaBias;
        beta = new FullyConnectedLayer(alpha.inputDims, alpha.outputDims).set(() -> {
          return initialFuzzyCoeff * (FastRandom.INSTANCE.random() - 0.5);
        });
        betaBias = new BiasLayer(alphaBias.bias.length());
        copyState(alpha, beta);
        copyState(alphaBias, betaBias);
        mode = NodeMode.Bilinear;
        if (isSkipFuzzy()) {
          nextPhase();
        }
        break;
      }
      case Bilinear:
        head = null;
        alpha = new SigmoidTreeNetwork(alpha, alphaBias);
        if (skipChildStage()) {
          ((SigmoidTreeNetwork) alpha).nextPhase();
        }
        beta = new SigmoidTreeNetwork(beta, betaBias);
        if (skipChildStage()) {
          ((SigmoidTreeNetwork) beta).nextPhase();
        }
        mode = NodeMode.Final;
        break;
      case Final:
        @Nonnull final SigmoidTreeNetwork alpha = (SigmoidTreeNetwork) this.alpha;
        @Nonnull final SigmoidTreeNetwork beta = (SigmoidTreeNetwork) this.beta;
        alpha.nextPhase();
        beta.nextPhase();
        break;
    }
  }

  public boolean skipChildStage() {
    return skipChildStage;
  }

  public @Override
  @SuppressWarnings("unused")
  SigmoidTreeNetwork addRef() {
    return (SigmoidTreeNetwork) super.addRef();
  }

  public enum NodeMode {
    Bilinear, Final, Fuzzy, Linear
  }

}
