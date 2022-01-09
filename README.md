# perceiverIO-project
# Perceiver IO

## Key ideas

Perceiver IO is an upgrade from original paper Perceiver, which is basically a stacked transformer network capable of taking various types of input while scaling linearly. 
Original Perceiver version could only have simple outputs such as classification scores etc.

Perceiver IO achieves strong results on tasks with highly structured output spaces.


__IDEA:__
Use a transformer to map input space to fixed size latent space.
Use a deep fully attentional network to perform task.

__COMPREHENSION:__
For each output value, we produce a query vector for this value, using our latent space refined by a FNetwork we produce a key vector created by our model for the task at hand.

\callout{
     For example if we wanted the model to predict optical flow on one particular pixel we could compose a query from the pixel’s xy coordinates plus an optical flow
    task embedding: the model would then attend using the query and produce a single flow vector
}

__Result__
We can produce many outputs with arbitrary shape and structure.

__Leverage__

Transformers - which leverage domain agnostic primitives for nonlocal processing of inputs.
Encoder-decoder - used in high-bandwith domains.

We decouple input size and latent size from output size.

## New contribution.

The cross-attention mechanism to map latents to arbitrarly size and structured outputs
    => Querying system.

uses attention over inputs and outputs of different sizes in part to produce an architecture that’s
more efficient than Transformers.

Cross-attention itself is in widespread use as a mechanism for allowing information from a source to
influence processing of a target domain of a different size or structure 

## The decoding stage for shape agnostic outputs.

What we basically do is :
We create a Query vector with same shape as output shape.
We use it to query the latent array we have refined so far.

## Architecture

### Input encoding.
We have an input array x of size $M \times C$.
In a regular transformer, all Q, K, V are $M \times D$ matrices.  We use an MLP to project $M \times C$ to $M \times D$.

Compute attention : $A = QK^T$,  ($M \times D) * (D \times M)$ so a is $M \times M$.

Compute output: $O = A V$

Result : $O(M^2 + MD)$

__In perceiver:__

Only K and V are $M \times D$, since Q is $N \times D$ with N << M.

Compute attention : $A = QK^T$  $(M \times D) * (D \times N)$ so a is $M \ times N$

Compute output: $O = A^T V$ $N \ times D$

Result: $O(MN + ND) = O(MN)$

How is Q computed ? In a regular transformer encoder, Q is computed using an MLP on the input. In this model however, Q is computed from a latent array of same dimensions.

We initialize this latent Query vector in latent space with a truncated normal distribution with mean 0 and standard deviation 0.02 and truncation bounds [-2,2]. The goal of this Q value is to query the input space.


With these steps, we have encoded the input $x \in \R^{M \times C}$ to the latent space  $z \in \R^{N \times D}$. 

    => with an attention mechanism from the Query vector to the, Key, Value vector that are projection of input space in latent space using MLP.

We call this the cross-attention module.

### Refining the latent space

We next process the latent z by applying a a regular self attention module in the latent space.

### Repeat

We repeat cros-attention and self-attention modules

__Weight sharing__:

Another important aspect of the model is weight sharing across the common modules, i.e. cross-attention modules share weights and latent transformer blocks share weights. One can think of it like RNN, unrolled in depth rather than time. One important point to note is, the authors found the sharing of weights with the first cross-attention block led to instabilities in training, hence for cross-attends, the weight sharing after the first instance.


### Decoding

The decoder works similarly to the encoder, but instead of mapping
inputs to latents, it maps latents to outputs

We initialize the output query matrix O, which is of same size as output. $O \times E$.
    =>  The goal of this Q value is to query the output space.    

The last z is $\in \R^{N \times D}$

We project this latent vector to $N \times E}$ using MLP to get K, V.

Compute attention : $ A = QK^T $ ($O \times E) * (E \times N)$ so a is $O \ times N$
Compute output: $O = A V$ $O \ times E$
Result: $O(ON + OE) = O(ON)$ N << M & N << O.

How is the output querry array initialized ?
The output query array is initialized with the domain-specific information needed for prediction. Like, for classification, the query matrix can be a random matrix initialized and learned during training. But for multi-task/ multi-model structure queries (like video autoencoding), the model learns a separate query for each of the modalities. Some intuition about the types of queries used in different tasks is given in the figure below.


## Comments

* Doesn't really scale linearly.
* Perceiver IO uses attention non-homogeneously, first using
it to map inputs to a latent space, then using it to process in that latent space, and finally using it to map to an output space.

* However, in contrast to the latent spaces used elsewhere in vision (e.g. [67]) the latent does not explicitly share the structure (spatial or otherwise) of the inputs. To decode this information, we
query for it using cross-attention.
* As usual in Transformer-style architectures, we apply the MLP
independently to each element of the index dimension.

## Output query initilization

To capture the structure of the output space, we must make sure this query contains the appropriate information.

This means the information query of the byte should reflect
the downstream task, and ideally captures any structure needed in the outputs. This may include the
spatial position in an image or the position of an output word in a sequence.
