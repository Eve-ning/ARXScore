$
¿£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
¾
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.02unknown8"
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	È*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	È*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0

lstm/lstm_cell/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 *&
shared_namelstm/lstm_cell/kernel

)lstm/lstm_cell/kernel/Read/ReadVariableOpReadVariableOplstm/lstm_cell/kernel* 
_output_shapes
:
 *
dtype0

lstm/lstm_cell/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
È *0
shared_name!lstm/lstm_cell/recurrent_kernel

3lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOpReadVariableOplstm/lstm_cell/recurrent_kernel* 
_output_shapes
:
È *
dtype0

lstm/lstm_cell/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_namelstm/lstm_cell/bias
x
'lstm/lstm_cell/bias/Read/ReadVariableOpReadVariableOplstm/lstm_cell/bias*
_output_shapes	
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	È*$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	È*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0

Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:*
dtype0

Adam/lstm/lstm_cell/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 *-
shared_nameAdam/lstm/lstm_cell/kernel/m

0Adam/lstm/lstm_cell/kernel/m/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell/kernel/m* 
_output_shapes
:
 *
dtype0
ª
&Adam/lstm/lstm_cell/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
È *7
shared_name(&Adam/lstm/lstm_cell/recurrent_kernel/m
£
:Adam/lstm/lstm_cell/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp&Adam/lstm/lstm_cell/recurrent_kernel/m* 
_output_shapes
:
È *
dtype0

Adam/lstm/lstm_cell/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameAdam/lstm/lstm_cell/bias/m

.Adam/lstm/lstm_cell/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell/bias/m*
_output_shapes	
: *
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	È*$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:	È*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0

Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:*
dtype0

Adam/lstm/lstm_cell/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
 *-
shared_nameAdam/lstm/lstm_cell/kernel/v

0Adam/lstm/lstm_cell/kernel/v/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell/kernel/v* 
_output_shapes
:
 *
dtype0
ª
&Adam/lstm/lstm_cell/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
È *7
shared_name(&Adam/lstm/lstm_cell/recurrent_kernel/v
£
:Adam/lstm/lstm_cell/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp&Adam/lstm/lstm_cell/recurrent_kernel/v* 
_output_shapes
:
È *
dtype0

Adam/lstm/lstm_cell/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_nameAdam/lstm/lstm_cell/bias/v

.Adam/lstm/lstm_cell/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm/lstm_cell/bias/v*
_output_shapes	
: *
dtype0

NoOpNoOp
0
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Æ/
value¼/B¹/ B²/
¥
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
#_self_saveable_object_factories
	optimizer

signatures
	trainable_variables

regularization_losses
	variables
	keras_api

cell

state_spec
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
w
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api


kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
w
# _self_saveable_object_factories
!regularization_losses
"	variables
#trainable_variables
$	keras_api


%kernel
&bias
#'_self_saveable_object_factories
(regularization_losses
)	variables
*trainable_variables
+	keras_api
 
¾
,iter

-beta_1

.beta_2
	/decay
0learning_ratembmc%md&me1mf2mg3mhvivj%vk&vl1vm2vn3vo
 
1
10
21
32
3
4
%5
&6
 
1
10
21
32
3
4
%5
&6
­
4layer_regularization_losses

5layers
	trainable_variables

regularization_losses
	variables
6non_trainable_variables
7metrics
8layer_metrics
£

1kernel
2recurrent_kernel
3bias
#9_self_saveable_object_factories
:regularization_losses
;	variables
<trainable_variables
=	keras_api
 
 
 

10
21
32

10
21
32
¹
>layer_regularization_losses

?layers
regularization_losses
	variables
trainable_variables
@non_trainable_variables
Ametrics

Bstates
Clayer_metrics
 
 
 
 
­
Dlayer_regularization_losses

Elayers
regularization_losses
	variables
trainable_variables
Fnon_trainable_variables
Gmetrics
Hlayer_metrics
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1

0
1
­
Ilayer_regularization_losses

Jlayers
regularization_losses
	variables
trainable_variables
Knon_trainable_variables
Lmetrics
Mlayer_metrics
 
 
 
 
­
Nlayer_regularization_losses

Olayers
!regularization_losses
"	variables
#trainable_variables
Pnon_trainable_variables
Qmetrics
Rlayer_metrics
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 
 

%0
&1

%0
&1
­
Slayer_regularization_losses

Tlayers
(regularization_losses
)	variables
*trainable_variables
Unon_trainable_variables
Vmetrics
Wlayer_metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUElstm/lstm_cell/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUElstm/lstm_cell/recurrent_kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUElstm/lstm_cell/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
 
#
0
1
2
3
4
 

X0
 
 
 

10
21
32

10
21
32
­
Ylayer_regularization_losses

Zlayers
:regularization_losses
;	variables
<trainable_variables
[non_trainable_variables
\metrics
]layer_metrics
 

0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	^total
	_count
`	variables
a	keras_api
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

^0
_1

`	variables
{y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/lstm/lstm_cell/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&Adam/lstm/lstm_cell/recurrent_kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/lstm/lstm_cell/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/lstm/lstm_cell/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE&Adam/lstm/lstm_cell/recurrent_kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/lstm/lstm_cell/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE

serving_default_lstm_inputPlaceholder*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0**
shape!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Ô
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_inputlstm/lstm_cell/kernellstm/lstm_cell/biaslstm/lstm_cell/recurrent_kerneldense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference_signature_wrapper_11697
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ª
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp)lstm/lstm_cell/kernel/Read/ReadVariableOp3lstm/lstm_cell/recurrent_kernel/Read/ReadVariableOp'lstm/lstm_cell/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp0Adam/lstm/lstm_cell/kernel/m/Read/ReadVariableOp:Adam/lstm/lstm_cell/recurrent_kernel/m/Read/ReadVariableOp.Adam/lstm/lstm_cell/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp0Adam/lstm/lstm_cell/kernel/v/Read/ReadVariableOp:Adam/lstm/lstm_cell/recurrent_kernel/v/Read/ReadVariableOp.Adam/lstm/lstm_cell/bias/v/Read/ReadVariableOpConst*)
Tin"
 2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_save_13970
õ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm/lstm_cell/kernellstm/lstm_cell/recurrent_kernellstm/lstm_cell/biastotalcountAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/lstm/lstm_cell/kernel/m&Adam/lstm/lstm_cell/recurrent_kernel/mAdam/lstm/lstm_cell/bias/mAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/lstm/lstm_cell/kernel/v&Adam/lstm/lstm_cell/recurrent_kernel/vAdam/lstm/lstm_cell/bias/v*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_restore_14064!

ù
sequential_lstm_while_cond_9912<
8sequential_lstm_while_sequential_lstm_while_loop_counterB
>sequential_lstm_while_sequential_lstm_while_maximum_iterations%
!sequential_lstm_while_placeholder'
#sequential_lstm_while_placeholder_1'
#sequential_lstm_while_placeholder_2'
#sequential_lstm_while_placeholder_3>
:sequential_lstm_while_less_sequential_lstm_strided_slice_1R
Nsequential_lstm_while_sequential_lstm_while_cond_9912___redundant_placeholder0R
Nsequential_lstm_while_sequential_lstm_while_cond_9912___redundant_placeholder1R
Nsequential_lstm_while_sequential_lstm_while_cond_9912___redundant_placeholder2R
Nsequential_lstm_while_sequential_lstm_while_cond_9912___redundant_placeholder3"
sequential_lstm_while_identity
À
sequential/lstm/while/LessLess!sequential_lstm_while_placeholder:sequential_lstm_while_less_sequential_lstm_strided_slice_1*
T0*
_output_shapes
: 2
sequential/lstm/while/Less
sequential/lstm/while/IdentityIdentitysequential/lstm/while/Less:z:0*
T0
*
_output_shapes
: 2 
sequential/lstm/while/Identity"I
sequential_lstm_while_identity'sequential/lstm/while/Identity:output:0*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
:
ñ
a
B__inference_dropout_layer_call_and_return_conditional_losses_13527

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÂ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÌ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
dropout/Mul_1s
IdentityIdentitydropout/Mul_1:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
¡±
¢

sequential_lstm_while_body_9913<
8sequential_lstm_while_sequential_lstm_while_loop_counterB
>sequential_lstm_while_sequential_lstm_while_maximum_iterations%
!sequential_lstm_while_placeholder'
#sequential_lstm_while_placeholder_1'
#sequential_lstm_while_placeholder_2'
#sequential_lstm_while_placeholder_3;
7sequential_lstm_while_sequential_lstm_strided_slice_1_0w
ssequential_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor_0C
?sequential_lstm_while_lstm_cell_split_readvariableop_resource_0E
Asequential_lstm_while_lstm_cell_split_1_readvariableop_resource_0=
9sequential_lstm_while_lstm_cell_readvariableop_resource_0"
sequential_lstm_while_identity$
 sequential_lstm_while_identity_1$
 sequential_lstm_while_identity_2$
 sequential_lstm_while_identity_3$
 sequential_lstm_while_identity_4$
 sequential_lstm_while_identity_59
5sequential_lstm_while_sequential_lstm_strided_slice_1u
qsequential_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensorA
=sequential_lstm_while_lstm_cell_split_readvariableop_resourceC
?sequential_lstm_while_lstm_cell_split_1_readvariableop_resource;
7sequential_lstm_while_lstm_cell_readvariableop_resourceã
Gsequential/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2I
Gsequential/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape´
9sequential/lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemssequential_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor_0!sequential_lstm_while_placeholderPsequential/lstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02;
9sequential/lstm/while/TensorArrayV2Read/TensorListGetItem
%sequential/lstm/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2'
%sequential/lstm/while/lstm_cell/Const¤
/sequential/lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :21
/sequential/lstm/while/lstm_cell/split/split_dimî
4sequential/lstm/while/lstm_cell/split/ReadVariableOpReadVariableOp?sequential_lstm_while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
 *
dtype026
4sequential/lstm/while/lstm_cell/split/ReadVariableOp¯
%sequential/lstm/while/lstm_cell/splitSplit8sequential/lstm/while/lstm_cell/split/split_dim:output:0<sequential/lstm/while/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È:
È:
È:
È*
	num_split2'
%sequential/lstm/while/lstm_cell/splitÿ
&sequential/lstm/while/lstm_cell/MatMulMatMul@sequential/lstm/while/TensorArrayV2Read/TensorListGetItem:item:0.sequential/lstm/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2(
&sequential/lstm/while/lstm_cell/MatMul
(sequential/lstm/while/lstm_cell/MatMul_1MatMul@sequential/lstm/while/TensorArrayV2Read/TensorListGetItem:item:0.sequential/lstm/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2*
(sequential/lstm/while/lstm_cell/MatMul_1
(sequential/lstm/while/lstm_cell/MatMul_2MatMul@sequential/lstm/while/TensorArrayV2Read/TensorListGetItem:item:0.sequential/lstm/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2*
(sequential/lstm/while/lstm_cell/MatMul_2
(sequential/lstm/while/lstm_cell/MatMul_3MatMul@sequential/lstm/while/TensorArrayV2Read/TensorListGetItem:item:0.sequential/lstm/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2*
(sequential/lstm/while/lstm_cell/MatMul_3
'sequential/lstm/while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2)
'sequential/lstm/while/lstm_cell/Const_1¨
1sequential/lstm/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 23
1sequential/lstm/while/lstm_cell/split_1/split_dimï
6sequential/lstm/while/lstm_cell/split_1/ReadVariableOpReadVariableOpAsequential_lstm_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
: *
dtype028
6sequential/lstm/while/lstm_cell/split_1/ReadVariableOp£
'sequential/lstm/while/lstm_cell/split_1Split:sequential/lstm/while/lstm_cell/split_1/split_dim:output:0>sequential/lstm/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:È:È:È:È*
	num_split2)
'sequential/lstm/while/lstm_cell/split_1ô
'sequential/lstm/while/lstm_cell/BiasAddBiasAdd0sequential/lstm/while/lstm_cell/MatMul:product:00sequential/lstm/while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2)
'sequential/lstm/while/lstm_cell/BiasAddú
)sequential/lstm/while/lstm_cell/BiasAdd_1BiasAdd2sequential/lstm/while/lstm_cell/MatMul_1:product:00sequential/lstm/while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2+
)sequential/lstm/while/lstm_cell/BiasAdd_1ú
)sequential/lstm/while/lstm_cell/BiasAdd_2BiasAdd2sequential/lstm/while/lstm_cell/MatMul_2:product:00sequential/lstm/while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2+
)sequential/lstm/while/lstm_cell/BiasAdd_2ú
)sequential/lstm/while/lstm_cell/BiasAdd_3BiasAdd2sequential/lstm/while/lstm_cell/MatMul_3:product:00sequential/lstm/while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2+
)sequential/lstm/while/lstm_cell/BiasAdd_3Ü
.sequential/lstm/while/lstm_cell/ReadVariableOpReadVariableOp9sequential_lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
È *
dtype020
.sequential/lstm/while/lstm_cell/ReadVariableOp»
3sequential/lstm/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        25
3sequential/lstm/while/lstm_cell/strided_slice/stack¿
5sequential/lstm/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    È   27
5sequential/lstm/while/lstm_cell/strided_slice/stack_1¿
5sequential/lstm/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      27
5sequential/lstm/while/lstm_cell/strided_slice/stack_2¾
-sequential/lstm/while/lstm_cell/strided_sliceStridedSlice6sequential/lstm/while/lstm_cell/ReadVariableOp:value:0<sequential/lstm/while/lstm_cell/strided_slice/stack:output:0>sequential/lstm/while/lstm_cell/strided_slice/stack_1:output:0>sequential/lstm/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2/
-sequential/lstm/while/lstm_cell/strided_sliceî
(sequential/lstm/while/lstm_cell/MatMul_4MatMul#sequential_lstm_while_placeholder_26sequential/lstm/while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2*
(sequential/lstm/while/lstm_cell/MatMul_4ì
#sequential/lstm/while/lstm_cell/addAddV20sequential/lstm/while/lstm_cell/BiasAdd:output:02sequential/lstm/while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2%
#sequential/lstm/while/lstm_cell/add
'sequential/lstm/while/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2)
'sequential/lstm/while/lstm_cell/Const_2
'sequential/lstm/while/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'sequential/lstm/while/lstm_cell/Const_3ß
#sequential/lstm/while/lstm_cell/MulMul'sequential/lstm/while/lstm_cell/add:z:00sequential/lstm/while/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2%
#sequential/lstm/while/lstm_cell/Mulã
%sequential/lstm/while/lstm_cell/Add_1Add'sequential/lstm/while/lstm_cell/Mul:z:00sequential/lstm/while/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2'
%sequential/lstm/while/lstm_cell/Add_1·
7sequential/lstm/while/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?29
7sequential/lstm/while/lstm_cell/clip_by_value/Minimum/y
5sequential/lstm/while/lstm_cell/clip_by_value/MinimumMinimum)sequential/lstm/while/lstm_cell/Add_1:z:0@sequential/lstm/while/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ27
5sequential/lstm/while/lstm_cell/clip_by_value/Minimum§
/sequential/lstm/while/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    21
/sequential/lstm/while/lstm_cell/clip_by_value/y
-sequential/lstm/while/lstm_cell/clip_by_valueMaximum9sequential/lstm/while/lstm_cell/clip_by_value/Minimum:z:08sequential/lstm/while/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2/
-sequential/lstm/while/lstm_cell/clip_by_valueà
0sequential/lstm/while/lstm_cell/ReadVariableOp_1ReadVariableOp9sequential_lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
È *
dtype022
0sequential/lstm/while/lstm_cell/ReadVariableOp_1¿
5sequential/lstm/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    È   27
5sequential/lstm/while/lstm_cell/strided_slice_1/stackÃ
7sequential/lstm/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      29
7sequential/lstm/while/lstm_cell/strided_slice_1/stack_1Ã
7sequential/lstm/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7sequential/lstm/while/lstm_cell/strided_slice_1/stack_2Ê
/sequential/lstm/while/lstm_cell/strided_slice_1StridedSlice8sequential/lstm/while/lstm_cell/ReadVariableOp_1:value:0>sequential/lstm/while/lstm_cell/strided_slice_1/stack:output:0@sequential/lstm/while/lstm_cell/strided_slice_1/stack_1:output:0@sequential/lstm/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask21
/sequential/lstm/while/lstm_cell/strided_slice_1ð
(sequential/lstm/while/lstm_cell/MatMul_5MatMul#sequential_lstm_while_placeholder_28sequential/lstm/while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2*
(sequential/lstm/while/lstm_cell/MatMul_5ò
%sequential/lstm/while/lstm_cell/add_2AddV22sequential/lstm/while/lstm_cell/BiasAdd_1:output:02sequential/lstm/while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2'
%sequential/lstm/while/lstm_cell/add_2
'sequential/lstm/while/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2)
'sequential/lstm/while/lstm_cell/Const_4
'sequential/lstm/while/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'sequential/lstm/while/lstm_cell/Const_5å
%sequential/lstm/while/lstm_cell/Mul_1Mul)sequential/lstm/while/lstm_cell/add_2:z:00sequential/lstm/while/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2'
%sequential/lstm/while/lstm_cell/Mul_1å
%sequential/lstm/while/lstm_cell/Add_3Add)sequential/lstm/while/lstm_cell/Mul_1:z:00sequential/lstm/while/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2'
%sequential/lstm/while/lstm_cell/Add_3»
9sequential/lstm/while/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2;
9sequential/lstm/while/lstm_cell/clip_by_value_1/Minimum/y
7sequential/lstm/while/lstm_cell/clip_by_value_1/MinimumMinimum)sequential/lstm/while/lstm_cell/Add_3:z:0Bsequential/lstm/while/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ29
7sequential/lstm/while/lstm_cell/clip_by_value_1/Minimum«
1sequential/lstm/while/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    23
1sequential/lstm/while/lstm_cell/clip_by_value_1/y
/sequential/lstm/while/lstm_cell/clip_by_value_1Maximum;sequential/lstm/while/lstm_cell/clip_by_value_1/Minimum:z:0:sequential/lstm/while/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ21
/sequential/lstm/while/lstm_cell/clip_by_value_1â
%sequential/lstm/while/lstm_cell/mul_2Mul3sequential/lstm/while/lstm_cell/clip_by_value_1:z:0#sequential_lstm_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2'
%sequential/lstm/while/lstm_cell/mul_2à
0sequential/lstm/while/lstm_cell/ReadVariableOp_2ReadVariableOp9sequential_lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
È *
dtype022
0sequential/lstm/while/lstm_cell/ReadVariableOp_2¿
5sequential/lstm/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"      27
5sequential/lstm/while/lstm_cell/strided_slice_2/stackÃ
7sequential/lstm/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  29
7sequential/lstm/while/lstm_cell/strided_slice_2/stack_1Ã
7sequential/lstm/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7sequential/lstm/while/lstm_cell/strided_slice_2/stack_2Ê
/sequential/lstm/while/lstm_cell/strided_slice_2StridedSlice8sequential/lstm/while/lstm_cell/ReadVariableOp_2:value:0>sequential/lstm/while/lstm_cell/strided_slice_2/stack:output:0@sequential/lstm/while/lstm_cell/strided_slice_2/stack_1:output:0@sequential/lstm/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask21
/sequential/lstm/while/lstm_cell/strided_slice_2ð
(sequential/lstm/while/lstm_cell/MatMul_6MatMul#sequential_lstm_while_placeholder_28sequential/lstm/while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2*
(sequential/lstm/while/lstm_cell/MatMul_6ò
%sequential/lstm/while/lstm_cell/add_4AddV22sequential/lstm/while/lstm_cell/BiasAdd_2:output:02sequential/lstm/while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2'
%sequential/lstm/while/lstm_cell/add_4²
$sequential/lstm/while/lstm_cell/TanhTanh)sequential/lstm/while/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2&
$sequential/lstm/while/lstm_cell/Tanhå
%sequential/lstm/while/lstm_cell/mul_3Mul1sequential/lstm/while/lstm_cell/clip_by_value:z:0(sequential/lstm/while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2'
%sequential/lstm/while/lstm_cell/mul_3à
%sequential/lstm/while/lstm_cell/add_5AddV2)sequential/lstm/while/lstm_cell/mul_2:z:0)sequential/lstm/while/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2'
%sequential/lstm/while/lstm_cell/add_5à
0sequential/lstm/while/lstm_cell/ReadVariableOp_3ReadVariableOp9sequential_lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
È *
dtype022
0sequential/lstm/while/lstm_cell/ReadVariableOp_3¿
5sequential/lstm/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  27
5sequential/lstm/while/lstm_cell/strided_slice_3/stackÃ
7sequential/lstm/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        29
7sequential/lstm/while/lstm_cell/strided_slice_3/stack_1Ã
7sequential/lstm/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      29
7sequential/lstm/while/lstm_cell/strided_slice_3/stack_2Ê
/sequential/lstm/while/lstm_cell/strided_slice_3StridedSlice8sequential/lstm/while/lstm_cell/ReadVariableOp_3:value:0>sequential/lstm/while/lstm_cell/strided_slice_3/stack:output:0@sequential/lstm/while/lstm_cell/strided_slice_3/stack_1:output:0@sequential/lstm/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask21
/sequential/lstm/while/lstm_cell/strided_slice_3ð
(sequential/lstm/while/lstm_cell/MatMul_7MatMul#sequential_lstm_while_placeholder_28sequential/lstm/while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2*
(sequential/lstm/while/lstm_cell/MatMul_7ò
%sequential/lstm/while/lstm_cell/add_6AddV22sequential/lstm/while/lstm_cell/BiasAdd_3:output:02sequential/lstm/while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2'
%sequential/lstm/while/lstm_cell/add_6
'sequential/lstm/while/lstm_cell/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2)
'sequential/lstm/while/lstm_cell/Const_6
'sequential/lstm/while/lstm_cell/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'sequential/lstm/while/lstm_cell/Const_7å
%sequential/lstm/while/lstm_cell/Mul_4Mul)sequential/lstm/while/lstm_cell/add_6:z:00sequential/lstm/while/lstm_cell/Const_6:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2'
%sequential/lstm/while/lstm_cell/Mul_4å
%sequential/lstm/while/lstm_cell/Add_7Add)sequential/lstm/while/lstm_cell/Mul_4:z:00sequential/lstm/while/lstm_cell/Const_7:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2'
%sequential/lstm/while/lstm_cell/Add_7»
9sequential/lstm/while/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2;
9sequential/lstm/while/lstm_cell/clip_by_value_2/Minimum/y
7sequential/lstm/while/lstm_cell/clip_by_value_2/MinimumMinimum)sequential/lstm/while/lstm_cell/Add_7:z:0Bsequential/lstm/while/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ29
7sequential/lstm/while/lstm_cell/clip_by_value_2/Minimum«
1sequential/lstm/while/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    23
1sequential/lstm/while/lstm_cell/clip_by_value_2/y
/sequential/lstm/while/lstm_cell/clip_by_value_2Maximum;sequential/lstm/while/lstm_cell/clip_by_value_2/Minimum:z:0:sequential/lstm/while/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ21
/sequential/lstm/while/lstm_cell/clip_by_value_2¶
&sequential/lstm/while/lstm_cell/Tanh_1Tanh)sequential/lstm/while/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2(
&sequential/lstm/while/lstm_cell/Tanh_1é
%sequential/lstm/while/lstm_cell/mul_5Mul3sequential/lstm/while/lstm_cell/clip_by_value_2:z:0*sequential/lstm/while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2'
%sequential/lstm/while/lstm_cell/mul_5­
:sequential/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#sequential_lstm_while_placeholder_1!sequential_lstm_while_placeholder)sequential/lstm/while/lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype02<
:sequential/lstm/while/TensorArrayV2Write/TensorListSetItem|
sequential/lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
sequential/lstm/while/add/y©
sequential/lstm/while/addAddV2!sequential_lstm_while_placeholder$sequential/lstm/while/add/y:output:0*
T0*
_output_shapes
: 2
sequential/lstm/while/add
sequential/lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
sequential/lstm/while/add_1/yÆ
sequential/lstm/while/add_1AddV28sequential_lstm_while_sequential_lstm_while_loop_counter&sequential/lstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2
sequential/lstm/while/add_1
sequential/lstm/while/IdentityIdentitysequential/lstm/while/add_1:z:0*
T0*
_output_shapes
: 2 
sequential/lstm/while/Identity±
 sequential/lstm/while/Identity_1Identity>sequential_lstm_while_sequential_lstm_while_maximum_iterations*
T0*
_output_shapes
: 2"
 sequential/lstm/while/Identity_1
 sequential/lstm/while/Identity_2Identitysequential/lstm/while/add:z:0*
T0*
_output_shapes
: 2"
 sequential/lstm/while/Identity_2½
 sequential/lstm/while/Identity_3IdentityJsequential/lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2"
 sequential/lstm/while/Identity_3®
 sequential/lstm/while/Identity_4Identity)sequential/lstm/while/lstm_cell/mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2"
 sequential/lstm/while/Identity_4®
 sequential/lstm/while/Identity_5Identity)sequential/lstm/while/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2"
 sequential/lstm/while/Identity_5"I
sequential_lstm_while_identity'sequential/lstm/while/Identity:output:0"M
 sequential_lstm_while_identity_1)sequential/lstm/while/Identity_1:output:0"M
 sequential_lstm_while_identity_2)sequential/lstm/while/Identity_2:output:0"M
 sequential_lstm_while_identity_3)sequential/lstm/while/Identity_3:output:0"M
 sequential_lstm_while_identity_4)sequential/lstm/while/Identity_4:output:0"M
 sequential_lstm_while_identity_5)sequential/lstm/while/Identity_5:output:0"t
7sequential_lstm_while_lstm_cell_readvariableop_resource9sequential_lstm_while_lstm_cell_readvariableop_resource_0"
?sequential_lstm_while_lstm_cell_split_1_readvariableop_resourceAsequential_lstm_while_lstm_cell_split_1_readvariableop_resource_0"
=sequential_lstm_while_lstm_cell_split_readvariableop_resource?sequential_lstm_while_lstm_cell_split_readvariableop_resource_0"p
5sequential_lstm_while_sequential_lstm_strided_slice_17sequential_lstm_while_sequential_lstm_strided_slice_1_0"è
qsequential_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensorssequential_lstm_while_tensorarrayv2read_tensorlistgetitem_sequential_lstm_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
: 
Í
ë
?__inference_lstm_layer_call_and_return_conditional_losses_12664
inputs_0+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identity¢whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimª
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
 *
dtype02 
lstm_cell/split/ReadVariableOp×
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È:
È:
È:
È*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim«
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
: *
dtype02"
 lstm_cell/split_1/ReadVariableOpË
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:È:È:È:È*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/BiasAdd¢
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/BiasAdd_1¢
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/BiasAdd_2¢
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/BiasAdd_3
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
È *
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    È   2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2º
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMulzeros:output:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/addk
lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
lstm_cell/Const_2k
lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/Const_3
lstm_cell/MulMullstm_cell/add:z:0lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Mul
lstm_cell/Add_1Addlstm_cell/Mul:z:0lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Add_1
!lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!lstm_cell/clip_by_value/Minimum/yÁ
lstm_cell/clip_by_value/MinimumMinimumlstm_cell/Add_1:z:0*lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2!
lstm_cell/clip_by_value/Minimum{
lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cell/clip_by_value/y¹
lstm_cell/clip_by_valueMaximum#lstm_cell/clip_by_value/Minimum:z:0"lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/clip_by_value
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
È *
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    È   2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Æ
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMulzeros:output:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_5
lstm_cell/add_2AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/add_2k
lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
lstm_cell/Const_4k
lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/Const_5
lstm_cell/Mul_1Mullstm_cell/add_2:z:0lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Mul_1
lstm_cell/Add_3Addlstm_cell/Mul_1:z:0lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Add_3
#lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2%
#lstm_cell/clip_by_value_1/Minimum/yÇ
!lstm_cell/clip_by_value_1/MinimumMinimumlstm_cell/Add_3:z:0,lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2#
!lstm_cell/clip_by_value_1/Minimum
lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cell/clip_by_value_1/yÁ
lstm_cell/clip_by_value_1Maximum%lstm_cell/clip_by_value_1/Minimum:z:0$lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/clip_by_value_1
lstm_cell/mul_2Mullstm_cell/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/mul_2
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
È *
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Æ
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMulzeros:output:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_6
lstm_cell/add_4AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/add_4p
lstm_cell/TanhTanhlstm_cell/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Tanh
lstm_cell/mul_3Mullstm_cell/clip_by_value:z:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/mul_3
lstm_cell/add_5AddV2lstm_cell/mul_2:z:0lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/add_5
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
È *
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Æ
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMulzeros:output:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_7
lstm_cell/add_6AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/add_6k
lstm_cell/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
lstm_cell/Const_6k
lstm_cell/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/Const_7
lstm_cell/Mul_4Mullstm_cell/add_6:z:0lstm_cell/Const_6:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Mul_4
lstm_cell/Add_7Addlstm_cell/Mul_4:z:0lstm_cell/Const_7:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Add_7
#lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2%
#lstm_cell/clip_by_value_2/Minimum/yÇ
!lstm_cell/clip_by_value_2/MinimumMinimumlstm_cell/Add_7:z:0,lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2#
!lstm_cell/clip_by_value_2/Minimum
lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cell/clip_by_value_2/yÁ
lstm_cell/clip_by_value_2Maximum%lstm_cell/clip_by_value_2/Minimum:z:0$lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/clip_by_value_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Tanh_1
lstm_cell/mul_5Mullstm_cell/clip_by_value_2:z:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/mul_5
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÝ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_12521*
condR
while_cond_12520*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimey
IdentityIdentitytranspose_1:y:0^while*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ª
¾
while_cond_13349
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_13349___redundant_placeholder03
/while_while_cond_13349___redundant_placeholder13
/while_while_cond_13349___redundant_placeholder23
/while_while_cond_13349___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
:
þ
·
E__inference_sequential_layer_call_and_return_conditional_losses_12034

inputs0
,lstm_lstm_cell_split_readvariableop_resource2
.lstm_lstm_cell_split_1_readvariableop_resource*
&lstm_lstm_cell_readvariableop_resource+
'dense_tensordot_readvariableop_resource)
%dense_biasadd_readvariableop_resource-
)dense_1_tensordot_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity¢
lstm/whileN

lstm/ShapeShapeinputs*
T0*
_output_shapes
:2

lstm/Shape~
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice/stack
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_1
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_2
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_sliceg
lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
lstm/zeros/mul/y
lstm/zeros/mulMullstm/strided_slice:output:0lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/muli
lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm/zeros/Less/y{
lstm/zeros/LessLesslstm/zeros/mul:z:0lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/Lessm
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
lstm/zeros/packed/1
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros/packedi
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros/Const

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

lstm/zerosk
lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
lstm/zeros_1/mul/y
lstm/zeros_1/mulMullstm/strided_slice:output:0lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/mulm
lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm/zeros_1/Less/y
lstm/zeros_1/LessLesslstm/zeros_1/mul:z:0lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/Lessq
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
lstm/zeros_1/packed/1
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros_1/packedm
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros_1/Const
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/zeros_1
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose/perm
lstm/transpose	Transposeinputslstm/transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
lstm/transpose^
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
:2
lstm/Shape_1
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_1/stack
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_1
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_2
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slice_1
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm/TensorArrayV2/element_shapeÆ
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2É
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2<
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shape
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02.
,lstm/TensorArrayUnstack/TensorListFromTensor
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_2/stack
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_1
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_2
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm/strided_slice_2n
lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/lstm_cell/Const
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm/lstm_cell/split/split_dim¹
#lstm/lstm_cell/split/ReadVariableOpReadVariableOp,lstm_lstm_cell_split_readvariableop_resource* 
_output_shapes
:
 *
dtype02%
#lstm/lstm_cell/split/ReadVariableOpë
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0+lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È:
È:
È:
È*
	num_split2
lstm/lstm_cell/split©
lstm/lstm_cell/MatMulMatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/MatMul­
lstm/lstm_cell/MatMul_1MatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/MatMul_1­
lstm/lstm_cell/MatMul_2MatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/MatMul_2­
lstm/lstm_cell/MatMul_3MatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/MatMul_3r
lstm/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm/lstm_cell/Const_1
 lstm/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 lstm/lstm_cell/split_1/split_dimº
%lstm/lstm_cell/split_1/ReadVariableOpReadVariableOp.lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
: *
dtype02'
%lstm/lstm_cell/split_1/ReadVariableOpß
lstm/lstm_cell/split_1Split)lstm/lstm_cell/split_1/split_dim:output:0-lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:È:È:È:È*
	num_split2
lstm/lstm_cell/split_1°
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/MatMul:product:0lstm/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/BiasAdd¶
lstm/lstm_cell/BiasAdd_1BiasAdd!lstm/lstm_cell/MatMul_1:product:0lstm/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/BiasAdd_1¶
lstm/lstm_cell/BiasAdd_2BiasAdd!lstm/lstm_cell/MatMul_2:product:0lstm/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/BiasAdd_2¶
lstm/lstm_cell/BiasAdd_3BiasAdd!lstm/lstm_cell/MatMul_3:product:0lstm/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/BiasAdd_3§
lstm/lstm_cell/ReadVariableOpReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
È *
dtype02
lstm/lstm_cell/ReadVariableOp
"lstm/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm/lstm_cell/strided_slice/stack
$lstm/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    È   2&
$lstm/lstm_cell/strided_slice/stack_1
$lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm/lstm_cell/strided_slice/stack_2Ø
lstm/lstm_cell/strided_sliceStridedSlice%lstm/lstm_cell/ReadVariableOp:value:0+lstm/lstm_cell/strided_slice/stack:output:0-lstm/lstm_cell/strided_slice/stack_1:output:0-lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
lstm/lstm_cell/strided_slice«
lstm/lstm_cell/MatMul_4MatMullstm/zeros:output:0%lstm/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/MatMul_4¨
lstm/lstm_cell/addAddV2lstm/lstm_cell/BiasAdd:output:0!lstm/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/addu
lstm/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
lstm/lstm_cell/Const_2u
lstm/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm/lstm_cell/Const_3
lstm/lstm_cell/MulMullstm/lstm_cell/add:z:0lstm/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/Mul
lstm/lstm_cell/Add_1Addlstm/lstm_cell/Mul:z:0lstm/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/Add_1
&lstm/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2(
&lstm/lstm_cell/clip_by_value/Minimum/yÕ
$lstm/lstm_cell/clip_by_value/MinimumMinimumlstm/lstm_cell/Add_1:z:0/lstm/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2&
$lstm/lstm_cell/clip_by_value/Minimum
lstm/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
lstm/lstm_cell/clip_by_value/yÍ
lstm/lstm_cell/clip_by_valueMaximum(lstm/lstm_cell/clip_by_value/Minimum:z:0'lstm/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/clip_by_value«
lstm/lstm_cell/ReadVariableOp_1ReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
È *
dtype02!
lstm/lstm_cell/ReadVariableOp_1
$lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    È   2&
$lstm/lstm_cell/strided_slice_1/stack¡
&lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2(
&lstm/lstm_cell/strided_slice_1/stack_1¡
&lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&lstm/lstm_cell/strided_slice_1/stack_2ä
lstm/lstm_cell/strided_slice_1StridedSlice'lstm/lstm_cell/ReadVariableOp_1:value:0-lstm/lstm_cell/strided_slice_1/stack:output:0/lstm/lstm_cell/strided_slice_1/stack_1:output:0/lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2 
lstm/lstm_cell/strided_slice_1­
lstm/lstm_cell/MatMul_5MatMullstm/zeros:output:0'lstm/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/MatMul_5®
lstm/lstm_cell/add_2AddV2!lstm/lstm_cell/BiasAdd_1:output:0!lstm/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/add_2u
lstm/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
lstm/lstm_cell/Const_4u
lstm/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm/lstm_cell/Const_5¡
lstm/lstm_cell/Mul_1Mullstm/lstm_cell/add_2:z:0lstm/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/Mul_1¡
lstm/lstm_cell/Add_3Addlstm/lstm_cell/Mul_1:z:0lstm/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/Add_3
(lstm/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2*
(lstm/lstm_cell/clip_by_value_1/Minimum/yÛ
&lstm/lstm_cell/clip_by_value_1/MinimumMinimumlstm/lstm_cell/Add_3:z:01lstm/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2(
&lstm/lstm_cell/clip_by_value_1/Minimum
 lstm/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 lstm/lstm_cell/clip_by_value_1/yÕ
lstm/lstm_cell/clip_by_value_1Maximum*lstm/lstm_cell/clip_by_value_1/Minimum:z:0)lstm/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2 
lstm/lstm_cell/clip_by_value_1¡
lstm/lstm_cell/mul_2Mul"lstm/lstm_cell/clip_by_value_1:z:0lstm/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/mul_2«
lstm/lstm_cell/ReadVariableOp_2ReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
È *
dtype02!
lstm/lstm_cell/ReadVariableOp_2
$lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm/lstm_cell/strided_slice_2/stack¡
&lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2(
&lstm/lstm_cell/strided_slice_2/stack_1¡
&lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&lstm/lstm_cell/strided_slice_2/stack_2ä
lstm/lstm_cell/strided_slice_2StridedSlice'lstm/lstm_cell/ReadVariableOp_2:value:0-lstm/lstm_cell/strided_slice_2/stack:output:0/lstm/lstm_cell/strided_slice_2/stack_1:output:0/lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2 
lstm/lstm_cell/strided_slice_2­
lstm/lstm_cell/MatMul_6MatMullstm/zeros:output:0'lstm/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/MatMul_6®
lstm/lstm_cell/add_4AddV2!lstm/lstm_cell/BiasAdd_2:output:0!lstm/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/add_4
lstm/lstm_cell/TanhTanhlstm/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/Tanh¡
lstm/lstm_cell/mul_3Mul lstm/lstm_cell/clip_by_value:z:0lstm/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/mul_3
lstm/lstm_cell/add_5AddV2lstm/lstm_cell/mul_2:z:0lstm/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/add_5«
lstm/lstm_cell/ReadVariableOp_3ReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
È *
dtype02!
lstm/lstm_cell/ReadVariableOp_3
$lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2&
$lstm/lstm_cell/strided_slice_3/stack¡
&lstm/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&lstm/lstm_cell/strided_slice_3/stack_1¡
&lstm/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&lstm/lstm_cell/strided_slice_3/stack_2ä
lstm/lstm_cell/strided_slice_3StridedSlice'lstm/lstm_cell/ReadVariableOp_3:value:0-lstm/lstm_cell/strided_slice_3/stack:output:0/lstm/lstm_cell/strided_slice_3/stack_1:output:0/lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2 
lstm/lstm_cell/strided_slice_3­
lstm/lstm_cell/MatMul_7MatMullstm/zeros:output:0'lstm/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/MatMul_7®
lstm/lstm_cell/add_6AddV2!lstm/lstm_cell/BiasAdd_3:output:0!lstm/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/add_6u
lstm/lstm_cell/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
lstm/lstm_cell/Const_6u
lstm/lstm_cell/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm/lstm_cell/Const_7¡
lstm/lstm_cell/Mul_4Mullstm/lstm_cell/add_6:z:0lstm/lstm_cell/Const_6:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/Mul_4¡
lstm/lstm_cell/Add_7Addlstm/lstm_cell/Mul_4:z:0lstm/lstm_cell/Const_7:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/Add_7
(lstm/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2*
(lstm/lstm_cell/clip_by_value_2/Minimum/yÛ
&lstm/lstm_cell/clip_by_value_2/MinimumMinimumlstm/lstm_cell/Add_7:z:01lstm/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2(
&lstm/lstm_cell/clip_by_value_2/Minimum
 lstm/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 lstm/lstm_cell/clip_by_value_2/yÕ
lstm/lstm_cell/clip_by_value_2Maximum*lstm/lstm_cell/clip_by_value_2/Minimum:z:0)lstm/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2 
lstm/lstm_cell/clip_by_value_2
lstm/lstm_cell/Tanh_1Tanhlstm/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/Tanh_1¥
lstm/lstm_cell/mul_5Mul"lstm/lstm_cell/clip_by_value_2:z:0lstm/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/mul_5
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   2$
"lstm/TensorArrayV2_1/element_shapeÌ
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2_1X
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
	lstm/time
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
lstm/while/maximum_iterationst
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/while/loop_counter¨

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_lstm_cell_split_readvariableop_resource.lstm_lstm_cell_split_1_readvariableop_resource&lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *%
_read_only_resource_inputs
	
*!
bodyR
lstm_while_body_11823*!
condR
lstm_while_cond_11822*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *
parallel_iterations 2

lstm/while¿
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   27
5lstm/TensorArrayV2Stack/TensorListStack/element_shape
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ*
element_dtype02)
'lstm/TensorArrayV2Stack/TensorListStack
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm/strided_slice_3/stack
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_3/stack_1
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_3/stack_2¹
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask2
lstm/strided_slice_3
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose_1/permÃ
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
lstm/transpose_1p
lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/runtimes
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/dropout/Const§
dropout/dropout/MulMullstm/transpose_1:y:0dropout/dropout/Const:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
dropout/dropout/Mulr
dropout/dropout/ShapeShapelstm/transpose_1:y:0*
T0*
_output_shapes
:2
dropout/dropout/ShapeÚ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ*
dtype02.
,dropout/dropout/random_uniform/RandomUniform
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2 
dropout/dropout/GreaterEqual/yì
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
dropout/dropout/GreaterEqual¥
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
dropout/dropout/Cast¨
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
dropout/dropout/Mul_1©
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes
:	È*
dtype02 
dense/Tensordot/ReadVariableOpv
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/axes}
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense/Tensordot/freew
dense/Tensordot/ShapeShapedropout/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense/Tensordot/Shape
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axisï
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axisõ
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2_1x
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod|
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_1 
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod_1|
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat/axisÎ
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat¤
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stack¿
dense/Tensordot/transpose	Transposedropout/dropout/Mul_1:z:0dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
dense/Tensordot/transpose·
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense/Tensordot/Reshape¶
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/Tensordot/MatMul|
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/Const_2
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axisÛ
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1±
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense/Tensordot
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp¨
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense/BiasAddw
dropout_1/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout_1/dropout/Const®
dropout_1/dropout/MulMuldense/BiasAdd:output:0 dropout_1/dropout/Const:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout_1/dropout/Mulx
dropout_1/dropout/ShapeShapedense/BiasAdd:output:0*
T0*
_output_shapes
:2
dropout_1/dropout/Shapeß
.dropout_1/dropout/random_uniform/RandomUniformRandomUniform dropout_1/dropout/Shape:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype020
.dropout_1/dropout/random_uniform/RandomUniform
 dropout_1/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2"
 dropout_1/dropout/GreaterEqual/yó
dropout_1/dropout/GreaterEqualGreaterEqual7dropout_1/dropout/random_uniform/RandomUniform:output:0)dropout_1/dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2 
dropout_1/dropout/GreaterEqualª
dropout_1/dropout/CastCast"dropout_1/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout_1/dropout/Cast¯
dropout_1/dropout/Mul_1Muldropout_1/dropout/Mul:z:0dropout_1/dropout/Cast:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout_1/dropout/Mul_1®
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02"
 dense_1/Tensordot/ReadVariableOpz
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_1/Tensordot/axes
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_1/Tensordot/free}
dense_1/Tensordot/ShapeShapedropout_1/dropout/Mul_1:z:0*
T0*
_output_shapes
:2
dense_1/Tensordot/Shape
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_1/Tensordot/GatherV2/axisù
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_1/Tensordot/GatherV2
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_1/Tensordot/GatherV2_1/axisÿ
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_1/Tensordot/GatherV2_1|
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_1/Tensordot/Const 
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_1/Tensordot/Prod
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_1/Tensordot/Const_1¨
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_1/Tensordot/Prod_1
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_1/Tensordot/concat/axisØ
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/concat¬
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/stackÆ
dense_1/Tensordot/transpose	Transposedropout_1/dropout/Mul_1:z:0!dense_1/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_1/Tensordot/transpose¿
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_1/Tensordot/Reshape¾
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/Tensordot/MatMul
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_1/Tensordot/Const_2
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_1/Tensordot/concat_1/axiså
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/concat_1¹
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_1/Tensordot¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp°
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_1/BiasAdd
IdentityIdentitydense_1/BiasAdd:output:0^lstm/while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::::2

lstm/while
lstm/while:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ýP

D__inference_lstm_cell_layer_call_and_return_conditional_losses_13738

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
 *
dtype02
split/ReadVariableOp¯
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È:
È:
È:
È*
	num_split2
splite
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
MatMuli
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

MatMul_1i
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

MatMul_2i
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
: *
dtype02
split_1/ReadVariableOp£
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:È:È:È:È*
	num_split2	
split_1t
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2	
BiasAddz
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
	BiasAdd_1z
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
	BiasAdd_2z
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
	BiasAdd_3z
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
È *
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    È   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2þ
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
strided_slices
MatMul_4MatMulstates_0strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

MatMul_4l
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3_
MulMuladd:z:0Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Mulc
Add_1AddMul:z:0Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value/Minimum/y
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
clip_by_value~
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
È *
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    È   2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
strided_slice_1u
MatMul_5MatMulstates_0strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

MatMul_5r
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5e
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Mul_1e
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value_1/Minimum/y
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
clip_by_value_1g
mul_2Mulclip_by_value_1:z:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
mul_2~
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
È *
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
strided_slice_2u
MatMul_6MatMulstates_0strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

MatMul_6r
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
add_4R
TanhTanh	add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Tanhe
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
mul_3`
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
add_5~
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
È *
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
strided_slice_3u
MatMul_7MatMulstates_0strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

MatMul_7r
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7e
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Mul_4e
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value_2/Minimum/y
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
clip_by_value_2V
Tanh_1Tanh	add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Tanh_1i
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
mul_5^
IdentityIdentity	mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identityb

Identity_1Identity	mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity_1b

Identity_2Identity	add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*[
_input_shapesJ
H:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ::::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
"
_user_specified_name
states/1
Á	
¢
lstm_while_cond_11822&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3(
$lstm_while_less_lstm_strided_slice_1=
9lstm_while_lstm_while_cond_11822___redundant_placeholder0=
9lstm_while_lstm_while_cond_11822___redundant_placeholder1=
9lstm_while_lstm_while_cond_11822___redundant_placeholder2=
9lstm_while_lstm_while_cond_11822___redundant_placeholder3
lstm_while_identity

lstm/while/LessLesslstm_while_placeholder$lstm_while_less_lstm_strided_slice_1*
T0*
_output_shapes
: 2
lstm/while/Lessl
lstm/while/IdentityIdentitylstm/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm/while/Identity"3
lstm_while_identitylstm/while/Identity:output:0*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
:
Æ
E
)__inference_dropout_1_layer_call_fn_13608

inputs
identityÏ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_115002
PartitionedCally
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Å
é
?__inference_lstm_layer_call_and_return_conditional_losses_13493

inputs+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identity¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimª
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
 *
dtype02 
lstm_cell/split/ReadVariableOp×
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È:
È:
È:
È*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim«
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
: *
dtype02"
 lstm_cell/split_1/ReadVariableOpË
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:È:È:È:È*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/BiasAdd¢
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/BiasAdd_1¢
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/BiasAdd_2¢
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/BiasAdd_3
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
È *
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    È   2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2º
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMulzeros:output:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/addk
lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
lstm_cell/Const_2k
lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/Const_3
lstm_cell/MulMullstm_cell/add:z:0lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Mul
lstm_cell/Add_1Addlstm_cell/Mul:z:0lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Add_1
!lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!lstm_cell/clip_by_value/Minimum/yÁ
lstm_cell/clip_by_value/MinimumMinimumlstm_cell/Add_1:z:0*lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2!
lstm_cell/clip_by_value/Minimum{
lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cell/clip_by_value/y¹
lstm_cell/clip_by_valueMaximum#lstm_cell/clip_by_value/Minimum:z:0"lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/clip_by_value
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
È *
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    È   2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Æ
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMulzeros:output:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_5
lstm_cell/add_2AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/add_2k
lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
lstm_cell/Const_4k
lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/Const_5
lstm_cell/Mul_1Mullstm_cell/add_2:z:0lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Mul_1
lstm_cell/Add_3Addlstm_cell/Mul_1:z:0lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Add_3
#lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2%
#lstm_cell/clip_by_value_1/Minimum/yÇ
!lstm_cell/clip_by_value_1/MinimumMinimumlstm_cell/Add_3:z:0,lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2#
!lstm_cell/clip_by_value_1/Minimum
lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cell/clip_by_value_1/yÁ
lstm_cell/clip_by_value_1Maximum%lstm_cell/clip_by_value_1/Minimum:z:0$lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/clip_by_value_1
lstm_cell/mul_2Mullstm_cell/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/mul_2
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
È *
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Æ
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMulzeros:output:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_6
lstm_cell/add_4AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/add_4p
lstm_cell/TanhTanhlstm_cell/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Tanh
lstm_cell/mul_3Mullstm_cell/clip_by_value:z:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/mul_3
lstm_cell/add_5AddV2lstm_cell/mul_2:z:0lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/add_5
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
È *
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Æ
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMulzeros:output:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_7
lstm_cell/add_6AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/add_6k
lstm_cell/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
lstm_cell/Const_6k
lstm_cell/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/Const_7
lstm_cell/Mul_4Mullstm_cell/add_6:z:0lstm_cell/Const_6:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Mul_4
lstm_cell/Add_7Addlstm_cell/Mul_4:z:0lstm_cell/Const_7:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Add_7
#lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2%
#lstm_cell/clip_by_value_2/Minimum/yÇ
!lstm_cell/clip_by_value_2/MinimumMinimumlstm_cell/Add_7:z:0,lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2#
!lstm_cell/clip_by_value_2/Minimum
lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cell/clip_by_value_2/yÁ
lstm_cell/clip_by_value_2Maximum%lstm_cell/clip_by_value_2/Minimum:z:0$lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/clip_by_value_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Tanh_1
lstm_cell/mul_5Mullstm_cell/clip_by_value_2:z:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/mul_5
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÝ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_13350*
condR
while_cond_13349*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimey
IdentityIdentitytranspose_1:y:0^while*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«

$__inference_lstm_layer_call_fn_13515

inputs
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_113772
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¶
«
@__inference_dense_layer_call_and_return_conditional_losses_13572

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	È*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisÑ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAddq
IdentityIdentityBiasAdd:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:::] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
ª
¾
while_cond_12520
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_12520___redundant_placeholder03
/while_while_cond_12520___redundant_placeholder13
/while_while_cond_12520___redundant_placeholder23
/while_while_cond_12520___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
:
´
­
B__inference_dense_1_layer_call_and_return_conditional_losses_13638

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisÑ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAddq
IdentityIdentityBiasAdd:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ë
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_13593

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÁ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yË
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1r
IdentityIdentitydropout/Mul_1:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
A

__inference__traced_save_13970
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop4
0savev2_lstm_lstm_cell_kernel_read_readvariableop>
:savev2_lstm_lstm_cell_recurrent_kernel_read_readvariableop2
.savev2_lstm_lstm_cell_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop;
7savev2_adam_lstm_lstm_cell_kernel_m_read_readvariableopE
Asavev2_adam_lstm_lstm_cell_recurrent_kernel_m_read_readvariableop9
5savev2_adam_lstm_lstm_cell_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop;
7savev2_adam_lstm_lstm_cell_kernel_v_read_readvariableopE
Asavev2_adam_lstm_lstm_cell_recurrent_kernel_v_read_readvariableop9
5savev2_adam_lstm_lstm_cell_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_2f53da15afe949e5a3f8da0d4ab5997e/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameÐ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*â
valueØBÕB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesÂ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop0savev2_lstm_lstm_cell_kernel_read_readvariableop:savev2_lstm_lstm_cell_recurrent_kernel_read_readvariableop.savev2_lstm_lstm_cell_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop7savev2_adam_lstm_lstm_cell_kernel_m_read_readvariableopAsavev2_adam_lstm_lstm_cell_recurrent_kernel_m_read_readvariableop5savev2_adam_lstm_lstm_cell_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop7savev2_adam_lstm_lstm_cell_kernel_v_read_readvariableopAsavev2_adam_lstm_lstm_cell_recurrent_kernel_v_read_readvariableop5savev2_adam_lstm_lstm_cell_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *+
dtypes!
2	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*ç
_input_shapesÕ
Ò: :	È:::: : : : : :
 :
È : : : :	È::::
 :
È : :	È::::
 :
È : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	È: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :&
"
 
_output_shapes
:
 :&"
 
_output_shapes
:
È :!

_output_shapes	
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	È: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::&"
 
_output_shapes
:
 :&"
 
_output_shapes
:
È :!

_output_shapes	
: :%!

_output_shapes
:	È: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::&"
 
_output_shapes
:
 :&"
 
_output_shapes
:
È :!

_output_shapes	
: :

_output_shapes
: 
«x
ü
!__inference__traced_restore_14064
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias%
!assignvariableop_2_dense_1_kernel#
assignvariableop_3_dense_1_bias 
assignvariableop_4_adam_iter"
assignvariableop_5_adam_beta_1"
assignvariableop_6_adam_beta_2!
assignvariableop_7_adam_decay)
%assignvariableop_8_adam_learning_rate,
(assignvariableop_9_lstm_lstm_cell_kernel7
3assignvariableop_10_lstm_lstm_cell_recurrent_kernel+
'assignvariableop_11_lstm_lstm_cell_bias
assignvariableop_12_total
assignvariableop_13_count+
'assignvariableop_14_adam_dense_kernel_m)
%assignvariableop_15_adam_dense_bias_m-
)assignvariableop_16_adam_dense_1_kernel_m+
'assignvariableop_17_adam_dense_1_bias_m4
0assignvariableop_18_adam_lstm_lstm_cell_kernel_m>
:assignvariableop_19_adam_lstm_lstm_cell_recurrent_kernel_m2
.assignvariableop_20_adam_lstm_lstm_cell_bias_m+
'assignvariableop_21_adam_dense_kernel_v)
%assignvariableop_22_adam_dense_bias_v-
)assignvariableop_23_adam_dense_1_kernel_v+
'assignvariableop_24_adam_dense_1_bias_v4
0assignvariableop_25_adam_lstm_lstm_cell_kernel_v>
:assignvariableop_26_adam_lstm_lstm_cell_recurrent_kernel_v2
.assignvariableop_27_adam_lstm_lstm_cell_bias_v
identity_29¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Ö
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*â
valueØBÕB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesÈ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*M
valueDBBB B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices½
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesv
t:::::::::::::::::::::::::::::*+
dtypes!
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¢
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¦
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¤
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4¡
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5£
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6£
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¢
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8ª
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9­
AssignVariableOp_9AssignVariableOp(assignvariableop_9_lstm_lstm_cell_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10»
AssignVariableOp_10AssignVariableOp3assignvariableop_10_lstm_lstm_cell_recurrent_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¯
AssignVariableOp_11AssignVariableOp'assignvariableop_11_lstm_lstm_cell_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12¡
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¡
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14¯
AssignVariableOp_14AssignVariableOp'assignvariableop_14_adam_dense_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15­
AssignVariableOp_15AssignVariableOp%assignvariableop_15_adam_dense_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16±
AssignVariableOp_16AssignVariableOp)assignvariableop_16_adam_dense_1_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¯
AssignVariableOp_17AssignVariableOp'assignvariableop_17_adam_dense_1_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18¸
AssignVariableOp_18AssignVariableOp0assignvariableop_18_adam_lstm_lstm_cell_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19Â
AssignVariableOp_19AssignVariableOp:assignvariableop_19_adam_lstm_lstm_cell_recurrent_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20¶
AssignVariableOp_20AssignVariableOp.assignvariableop_20_adam_lstm_lstm_cell_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21¯
AssignVariableOp_21AssignVariableOp'assignvariableop_21_adam_dense_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22­
AssignVariableOp_22AssignVariableOp%assignvariableop_22_adam_dense_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23±
AssignVariableOp_23AssignVariableOp)assignvariableop_23_adam_dense_1_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24¯
AssignVariableOp_24AssignVariableOp'assignvariableop_24_adam_dense_1_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25¸
AssignVariableOp_25AssignVariableOp0assignvariableop_25_adam_lstm_lstm_cell_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26Â
AssignVariableOp_26AssignVariableOp:assignvariableop_26_adam_lstm_lstm_cell_recurrent_kernel_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27¶
AssignVariableOp_27AssignVariableOp.assignvariableop_27_adam_lstm_lstm_cell_bias_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_279
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpÆ
Identity_28Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_28¹
Identity_29IdentityIdentity_28:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_29"#
identity_29Identity_29:output:0*
_input_shapest
r: ::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ýP

D__inference_lstm_cell_layer_call_and_return_conditional_losses_13829

inputs
states_0
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
 *
dtype02
split/ReadVariableOp¯
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È:
È:
È:
È*
	num_split2
splite
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
MatMuli
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

MatMul_1i
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

MatMul_2i
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
: *
dtype02
split_1/ReadVariableOp£
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:È:È:È:È*
	num_split2	
split_1t
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2	
BiasAddz
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
	BiasAdd_1z
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
	BiasAdd_2z
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
	BiasAdd_3z
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
È *
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    È   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2þ
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
strided_slices
MatMul_4MatMulstates_0strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

MatMul_4l
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3_
MulMuladd:z:0Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Mulc
Add_1AddMul:z:0Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value/Minimum/y
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
clip_by_value~
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
È *
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    È   2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
strided_slice_1u
MatMul_5MatMulstates_0strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

MatMul_5r
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5e
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Mul_1e
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value_1/Minimum/y
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
clip_by_value_1g
mul_2Mulclip_by_value_1:z:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
mul_2~
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
È *
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
strided_slice_2u
MatMul_6MatMulstates_0strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

MatMul_6r
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
add_4R
TanhTanh	add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Tanhe
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
mul_3`
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
add_5~
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
È *
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
strided_slice_3u
MatMul_7MatMulstates_0strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

MatMul_7r
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7e
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Mul_4e
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value_2/Minimum/y
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
clip_by_value_2V
Tanh_1Tanh	add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Tanh_1i
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
mul_5^
IdentityIdentity	mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identityb

Identity_1Identity	mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity_1b

Identity_2Identity	add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*[
_input_shapesJ
H:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ::::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
"
_user_specified_name
states/1
¿

E__inference_sequential_layer_call_and_return_conditional_losses_11609

inputs

lstm_11589

lstm_11591

lstm_11593
dense_11597
dense_11599
dense_1_11603
dense_1_11605
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢lstm/StatefulPartitionedCall
lstm/StatefulPartitionedCallStatefulPartitionedCallinputs
lstm_11589
lstm_11591
lstm_11593*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_111082
lstm/StatefulPartitionedCall
dropout/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_114192!
dropout/StatefulPartitionedCall±
dense/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_11597dense_11599*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_114672
dense/StatefulPartitionedCall½
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_114952#
!dropout_1/StatefulPartitionedCall½
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_1_11603dense_1_11605*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_115432!
dense_1/StatefulPartitionedCall°
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall^lstm/StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
¾
while_cond_10625
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_10625___redundant_placeholder03
/while_while_cond_10625___redundant_placeholder13
/while_while_cond_10625___redundant_placeholder23
/while_while_cond_10625___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
:
Ð
Ó
while_body_13081
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
/while_lstm_cell_split_readvariableop_resource_05
1while_lstm_cell_split_1_readvariableop_resource_0-
)while_lstm_cell_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
-while_lstm_cell_split_readvariableop_resource3
/while_lstm_cell_split_1_readvariableop_resource+
'while_lstm_cell_readvariableop_resourceÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim¾
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
 *
dtype02&
$while/lstm_cell/split/ReadVariableOpï
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È:
È:
È:
È*
	num_split2
while/lstm_cell/split¿
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMulÃ
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_1Ã
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_2Ã
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_3t
while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const_1
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!while/lstm_cell/split_1/split_dim¿
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
: *
dtype02(
&while/lstm_cell/split_1/ReadVariableOpã
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:È:È:È:È*
	num_split2
while/lstm_cell/split_1´
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/BiasAddº
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/BiasAdd_1º
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/BiasAdd_2º
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/BiasAdd_3¬
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02 
while/lstm_cell/ReadVariableOp
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#while/lstm_cell/strided_slice/stack
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    È   2'
%while/lstm_cell/strided_slice/stack_1
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice/stack_2Þ
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
while/lstm_cell/strided_slice®
while/lstm_cell/MatMul_4MatMulwhile_placeholder_2&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_4¬
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/addw
while/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
while/lstm_cell/Const_2w
while/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/Const_3
while/lstm_cell/MulMulwhile/lstm_cell/add:z:0 while/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Mul£
while/lstm_cell/Add_1Addwhile/lstm_cell/Mul:z:0 while/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Add_1
'while/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'while/lstm_cell/clip_by_value/Minimum/yÙ
%while/lstm_cell/clip_by_value/MinimumMinimumwhile/lstm_cell/Add_1:z:00while/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2'
%while/lstm_cell/clip_by_value/Minimum
while/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
while/lstm_cell/clip_by_value/yÑ
while/lstm_cell/clip_by_valueMaximum)while/lstm_cell/clip_by_value/Minimum:z:0(while/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/clip_by_value°
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02"
 while/lstm_cell/ReadVariableOp_1
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    È   2'
%while/lstm_cell/strided_slice_1/stack£
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_1£
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_2ê
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_1°
while/lstm_cell/MatMul_5MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_5²
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/add_2w
while/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
while/lstm_cell/Const_4w
while/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/Const_5¥
while/lstm_cell/Mul_1Mulwhile/lstm_cell/add_2:z:0 while/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Mul_1¥
while/lstm_cell/Add_3Addwhile/lstm_cell/Mul_1:z:0 while/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Add_3
)while/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)while/lstm_cell/clip_by_value_1/Minimum/yß
'while/lstm_cell/clip_by_value_1/MinimumMinimumwhile/lstm_cell/Add_3:z:02while/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2)
'while/lstm_cell/clip_by_value_1/Minimum
!while/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!while/lstm_cell/clip_by_value_1/yÙ
while/lstm_cell/clip_by_value_1Maximum+while/lstm_cell/clip_by_value_1/Minimum:z:0*while/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2!
while/lstm_cell/clip_by_value_1¢
while/lstm_cell/mul_2Mul#while/lstm_cell/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/mul_2°
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02"
 while/lstm_cell/ReadVariableOp_2
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice_2/stack£
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2)
'while/lstm_cell/strided_slice_2/stack_1£
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_2/stack_2ê
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_2°
while/lstm_cell/MatMul_6MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_6²
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/add_4
while/lstm_cell/TanhTanhwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Tanh¥
while/lstm_cell/mul_3Mul!while/lstm_cell/clip_by_value:z:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/mul_3 
while/lstm_cell/add_5AddV2while/lstm_cell/mul_2:z:0while/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/add_5°
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02"
 while/lstm_cell/ReadVariableOp_3
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2'
%while/lstm_cell/strided_slice_3/stack£
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell/strided_slice_3/stack_1£
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_3/stack_2ê
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_3°
while/lstm_cell/MatMul_7MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_7²
while/lstm_cell/add_6AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/add_6w
while/lstm_cell/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
while/lstm_cell/Const_6w
while/lstm_cell/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/Const_7¥
while/lstm_cell/Mul_4Mulwhile/lstm_cell/add_6:z:0 while/lstm_cell/Const_6:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Mul_4¥
while/lstm_cell/Add_7Addwhile/lstm_cell/Mul_4:z:0 while/lstm_cell/Const_7:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Add_7
)while/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)while/lstm_cell/clip_by_value_2/Minimum/yß
'while/lstm_cell/clip_by_value_2/MinimumMinimumwhile/lstm_cell/Add_7:z:02while/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2)
'while/lstm_cell/clip_by_value_2/Minimum
!while/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!while/lstm_cell/clip_by_value_2/yÙ
while/lstm_cell/clip_by_value_2Maximum+while/lstm_cell/clip_by_value_2/Minimum:z:0*while/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2!
while/lstm_cell/clip_by_value_2
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Tanh_1©
while/lstm_cell/mul_5Mul#while/lstm_cell/clip_by_value_2:z:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/mul_5Ý
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3~
while/Identity_4Identitywhile/lstm_cell/mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/Identity_4~
while/Identity_5Identitywhile/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
: 
ý
`
B__inference_dropout_layer_call_and_return_conditional_losses_11424

inputs

identity_1h
IdentityIdentityinputs*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2

Identityw

Identity_1IdentityIdentity:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2

Identity_1"!

identity_1Identity_1:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
Å
é
?__inference_lstm_layer_call_and_return_conditional_losses_13224

inputs+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identity¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimª
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
 *
dtype02 
lstm_cell/split/ReadVariableOp×
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È:
È:
È:
È*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim«
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
: *
dtype02"
 lstm_cell/split_1/ReadVariableOpË
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:È:È:È:È*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/BiasAdd¢
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/BiasAdd_1¢
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/BiasAdd_2¢
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/BiasAdd_3
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
È *
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    È   2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2º
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMulzeros:output:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/addk
lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
lstm_cell/Const_2k
lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/Const_3
lstm_cell/MulMullstm_cell/add:z:0lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Mul
lstm_cell/Add_1Addlstm_cell/Mul:z:0lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Add_1
!lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!lstm_cell/clip_by_value/Minimum/yÁ
lstm_cell/clip_by_value/MinimumMinimumlstm_cell/Add_1:z:0*lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2!
lstm_cell/clip_by_value/Minimum{
lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cell/clip_by_value/y¹
lstm_cell/clip_by_valueMaximum#lstm_cell/clip_by_value/Minimum:z:0"lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/clip_by_value
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
È *
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    È   2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Æ
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMulzeros:output:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_5
lstm_cell/add_2AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/add_2k
lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
lstm_cell/Const_4k
lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/Const_5
lstm_cell/Mul_1Mullstm_cell/add_2:z:0lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Mul_1
lstm_cell/Add_3Addlstm_cell/Mul_1:z:0lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Add_3
#lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2%
#lstm_cell/clip_by_value_1/Minimum/yÇ
!lstm_cell/clip_by_value_1/MinimumMinimumlstm_cell/Add_3:z:0,lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2#
!lstm_cell/clip_by_value_1/Minimum
lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cell/clip_by_value_1/yÁ
lstm_cell/clip_by_value_1Maximum%lstm_cell/clip_by_value_1/Minimum:z:0$lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/clip_by_value_1
lstm_cell/mul_2Mullstm_cell/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/mul_2
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
È *
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Æ
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMulzeros:output:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_6
lstm_cell/add_4AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/add_4p
lstm_cell/TanhTanhlstm_cell/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Tanh
lstm_cell/mul_3Mullstm_cell/clip_by_value:z:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/mul_3
lstm_cell/add_5AddV2lstm_cell/mul_2:z:0lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/add_5
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
È *
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Æ
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMulzeros:output:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_7
lstm_cell/add_6AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/add_6k
lstm_cell/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
lstm_cell/Const_6k
lstm_cell/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/Const_7
lstm_cell/Mul_4Mullstm_cell/add_6:z:0lstm_cell/Const_6:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Mul_4
lstm_cell/Add_7Addlstm_cell/Mul_4:z:0lstm_cell/Const_7:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Add_7
#lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2%
#lstm_cell/clip_by_value_2/Minimum/yÇ
!lstm_cell/clip_by_value_2/MinimumMinimumlstm_cell/Add_7:z:0,lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2#
!lstm_cell/clip_by_value_2/Minimum
lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cell/clip_by_value_2/yÁ
lstm_cell/clip_by_value_2Maximum%lstm_cell/clip_by_value_2/Minimum:z:0$lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/clip_by_value_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Tanh_1
lstm_cell/mul_5Mullstm_cell/clip_by_value_2:z:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/mul_5
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÝ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_13081*
condR
while_cond_13080*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimey
IdentityIdentitytranspose_1:y:0^while*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¯ 
î
 __inference__wrapped_model_10110

lstm_input;
7sequential_lstm_lstm_cell_split_readvariableop_resource=
9sequential_lstm_lstm_cell_split_1_readvariableop_resource5
1sequential_lstm_lstm_cell_readvariableop_resource6
2sequential_dense_tensordot_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource8
4sequential_dense_1_tensordot_readvariableop_resource6
2sequential_dense_1_biasadd_readvariableop_resource
identity¢sequential/lstm/whileh
sequential/lstm/ShapeShape
lstm_input*
T0*
_output_shapes
:2
sequential/lstm/Shape
#sequential/lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#sequential/lstm/strided_slice/stack
%sequential/lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%sequential/lstm/strided_slice/stack_1
%sequential/lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%sequential/lstm/strided_slice/stack_2Â
sequential/lstm/strided_sliceStridedSlicesequential/lstm/Shape:output:0,sequential/lstm/strided_slice/stack:output:0.sequential/lstm/strided_slice/stack_1:output:0.sequential/lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
sequential/lstm/strided_slice}
sequential/lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
sequential/lstm/zeros/mul/y¬
sequential/lstm/zeros/mulMul&sequential/lstm/strided_slice:output:0$sequential/lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
sequential/lstm/zeros/mul
sequential/lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
sequential/lstm/zeros/Less/y§
sequential/lstm/zeros/LessLesssequential/lstm/zeros/mul:z:0%sequential/lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
sequential/lstm/zeros/Less
sequential/lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2 
sequential/lstm/zeros/packed/1Ã
sequential/lstm/zeros/packedPack&sequential/lstm/strided_slice:output:0'sequential/lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
sequential/lstm/zeros/packed
sequential/lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/lstm/zeros/Const¶
sequential/lstm/zerosFill%sequential/lstm/zeros/packed:output:0$sequential/lstm/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
sequential/lstm/zeros
sequential/lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
sequential/lstm/zeros_1/mul/y²
sequential/lstm/zeros_1/mulMul&sequential/lstm/strided_slice:output:0&sequential/lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
sequential/lstm/zeros_1/mul
sequential/lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2 
sequential/lstm/zeros_1/Less/y¯
sequential/lstm/zeros_1/LessLesssequential/lstm/zeros_1/mul:z:0'sequential/lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
sequential/lstm/zeros_1/Less
 sequential/lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2"
 sequential/lstm/zeros_1/packed/1É
sequential/lstm/zeros_1/packedPack&sequential/lstm/strided_slice:output:0)sequential/lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2 
sequential/lstm/zeros_1/packed
sequential/lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/lstm/zeros_1/Const¾
sequential/lstm/zeros_1Fill'sequential/lstm/zeros_1/packed:output:0&sequential/lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
sequential/lstm/zeros_1
sequential/lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2 
sequential/lstm/transpose/perm¸
sequential/lstm/transpose	Transpose
lstm_input'sequential/lstm/transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
sequential/lstm/transpose
sequential/lstm/Shape_1Shapesequential/lstm/transpose:y:0*
T0*
_output_shapes
:2
sequential/lstm/Shape_1
%sequential/lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential/lstm/strided_slice_1/stack
'sequential/lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'sequential/lstm/strided_slice_1/stack_1
'sequential/lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'sequential/lstm/strided_slice_1/stack_2Î
sequential/lstm/strided_slice_1StridedSlice sequential/lstm/Shape_1:output:0.sequential/lstm/strided_slice_1/stack:output:00sequential/lstm/strided_slice_1/stack_1:output:00sequential/lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
sequential/lstm/strided_slice_1¥
+sequential/lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2-
+sequential/lstm/TensorArrayV2/element_shapeò
sequential/lstm/TensorArrayV2TensorListReserve4sequential/lstm/TensorArrayV2/element_shape:output:0(sequential/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
sequential/lstm/TensorArrayV2ß
Esequential/lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2G
Esequential/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape¸
7sequential/lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsequential/lstm/transpose:y:0Nsequential/lstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type029
7sequential/lstm/TensorArrayUnstack/TensorListFromTensor
%sequential/lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%sequential/lstm/strided_slice_2/stack
'sequential/lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'sequential/lstm/strided_slice_2/stack_1
'sequential/lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'sequential/lstm/strided_slice_2/stack_2Ý
sequential/lstm/strided_slice_2StridedSlicesequential/lstm/transpose:y:0.sequential/lstm/strided_slice_2/stack:output:00sequential/lstm/strided_slice_2/stack_1:output:00sequential/lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2!
sequential/lstm/strided_slice_2
sequential/lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2!
sequential/lstm/lstm_cell/Const
)sequential/lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2+
)sequential/lstm/lstm_cell/split/split_dimÚ
.sequential/lstm/lstm_cell/split/ReadVariableOpReadVariableOp7sequential_lstm_lstm_cell_split_readvariableop_resource* 
_output_shapes
:
 *
dtype020
.sequential/lstm/lstm_cell/split/ReadVariableOp
sequential/lstm/lstm_cell/splitSplit2sequential/lstm/lstm_cell/split/split_dim:output:06sequential/lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È:
È:
È:
È*
	num_split2!
sequential/lstm/lstm_cell/splitÕ
 sequential/lstm/lstm_cell/MatMulMatMul(sequential/lstm/strided_slice_2:output:0(sequential/lstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2"
 sequential/lstm/lstm_cell/MatMulÙ
"sequential/lstm/lstm_cell/MatMul_1MatMul(sequential/lstm/strided_slice_2:output:0(sequential/lstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2$
"sequential/lstm/lstm_cell/MatMul_1Ù
"sequential/lstm/lstm_cell/MatMul_2MatMul(sequential/lstm/strided_slice_2:output:0(sequential/lstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2$
"sequential/lstm/lstm_cell/MatMul_2Ù
"sequential/lstm/lstm_cell/MatMul_3MatMul(sequential/lstm/strided_slice_2:output:0(sequential/lstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2$
"sequential/lstm/lstm_cell/MatMul_3
!sequential/lstm/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2#
!sequential/lstm/lstm_cell/Const_1
+sequential/lstm/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2-
+sequential/lstm/lstm_cell/split_1/split_dimÛ
0sequential/lstm/lstm_cell/split_1/ReadVariableOpReadVariableOp9sequential_lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
: *
dtype022
0sequential/lstm/lstm_cell/split_1/ReadVariableOp
!sequential/lstm/lstm_cell/split_1Split4sequential/lstm/lstm_cell/split_1/split_dim:output:08sequential/lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:È:È:È:È*
	num_split2#
!sequential/lstm/lstm_cell/split_1Ü
!sequential/lstm/lstm_cell/BiasAddBiasAdd*sequential/lstm/lstm_cell/MatMul:product:0*sequential/lstm/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2#
!sequential/lstm/lstm_cell/BiasAddâ
#sequential/lstm/lstm_cell/BiasAdd_1BiasAdd,sequential/lstm/lstm_cell/MatMul_1:product:0*sequential/lstm/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2%
#sequential/lstm/lstm_cell/BiasAdd_1â
#sequential/lstm/lstm_cell/BiasAdd_2BiasAdd,sequential/lstm/lstm_cell/MatMul_2:product:0*sequential/lstm/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2%
#sequential/lstm/lstm_cell/BiasAdd_2â
#sequential/lstm/lstm_cell/BiasAdd_3BiasAdd,sequential/lstm/lstm_cell/MatMul_3:product:0*sequential/lstm/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2%
#sequential/lstm/lstm_cell/BiasAdd_3È
(sequential/lstm/lstm_cell/ReadVariableOpReadVariableOp1sequential_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
È *
dtype02*
(sequential/lstm/lstm_cell/ReadVariableOp¯
-sequential/lstm/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2/
-sequential/lstm/lstm_cell/strided_slice/stack³
/sequential/lstm/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    È   21
/sequential/lstm/lstm_cell/strided_slice/stack_1³
/sequential/lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      21
/sequential/lstm/lstm_cell/strided_slice/stack_2
'sequential/lstm/lstm_cell/strided_sliceStridedSlice0sequential/lstm/lstm_cell/ReadVariableOp:value:06sequential/lstm/lstm_cell/strided_slice/stack:output:08sequential/lstm/lstm_cell/strided_slice/stack_1:output:08sequential/lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2)
'sequential/lstm/lstm_cell/strided_slice×
"sequential/lstm/lstm_cell/MatMul_4MatMulsequential/lstm/zeros:output:00sequential/lstm/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2$
"sequential/lstm/lstm_cell/MatMul_4Ô
sequential/lstm/lstm_cell/addAddV2*sequential/lstm/lstm_cell/BiasAdd:output:0,sequential/lstm/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
sequential/lstm/lstm_cell/add
!sequential/lstm/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2#
!sequential/lstm/lstm_cell/Const_2
!sequential/lstm/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!sequential/lstm/lstm_cell/Const_3Ç
sequential/lstm/lstm_cell/MulMul!sequential/lstm/lstm_cell/add:z:0*sequential/lstm/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
sequential/lstm/lstm_cell/MulË
sequential/lstm/lstm_cell/Add_1Add!sequential/lstm/lstm_cell/Mul:z:0*sequential/lstm/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2!
sequential/lstm/lstm_cell/Add_1«
1sequential/lstm/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?23
1sequential/lstm/lstm_cell/clip_by_value/Minimum/y
/sequential/lstm/lstm_cell/clip_by_value/MinimumMinimum#sequential/lstm/lstm_cell/Add_1:z:0:sequential/lstm/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ21
/sequential/lstm/lstm_cell/clip_by_value/Minimum
)sequential/lstm/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2+
)sequential/lstm/lstm_cell/clip_by_value/yù
'sequential/lstm/lstm_cell/clip_by_valueMaximum3sequential/lstm/lstm_cell/clip_by_value/Minimum:z:02sequential/lstm/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2)
'sequential/lstm/lstm_cell/clip_by_valueÌ
*sequential/lstm/lstm_cell/ReadVariableOp_1ReadVariableOp1sequential_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
È *
dtype02,
*sequential/lstm/lstm_cell/ReadVariableOp_1³
/sequential/lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    È   21
/sequential/lstm/lstm_cell/strided_slice_1/stack·
1sequential/lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      23
1sequential/lstm/lstm_cell/strided_slice_1/stack_1·
1sequential/lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1sequential/lstm/lstm_cell/strided_slice_1/stack_2¦
)sequential/lstm/lstm_cell/strided_slice_1StridedSlice2sequential/lstm/lstm_cell/ReadVariableOp_1:value:08sequential/lstm/lstm_cell/strided_slice_1/stack:output:0:sequential/lstm/lstm_cell/strided_slice_1/stack_1:output:0:sequential/lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2+
)sequential/lstm/lstm_cell/strided_slice_1Ù
"sequential/lstm/lstm_cell/MatMul_5MatMulsequential/lstm/zeros:output:02sequential/lstm/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2$
"sequential/lstm/lstm_cell/MatMul_5Ú
sequential/lstm/lstm_cell/add_2AddV2,sequential/lstm/lstm_cell/BiasAdd_1:output:0,sequential/lstm/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2!
sequential/lstm/lstm_cell/add_2
!sequential/lstm/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2#
!sequential/lstm/lstm_cell/Const_4
!sequential/lstm/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!sequential/lstm/lstm_cell/Const_5Í
sequential/lstm/lstm_cell/Mul_1Mul#sequential/lstm/lstm_cell/add_2:z:0*sequential/lstm/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2!
sequential/lstm/lstm_cell/Mul_1Í
sequential/lstm/lstm_cell/Add_3Add#sequential/lstm/lstm_cell/Mul_1:z:0*sequential/lstm/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2!
sequential/lstm/lstm_cell/Add_3¯
3sequential/lstm/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?25
3sequential/lstm/lstm_cell/clip_by_value_1/Minimum/y
1sequential/lstm/lstm_cell/clip_by_value_1/MinimumMinimum#sequential/lstm/lstm_cell/Add_3:z:0<sequential/lstm/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ23
1sequential/lstm/lstm_cell/clip_by_value_1/Minimum
+sequential/lstm/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+sequential/lstm/lstm_cell/clip_by_value_1/y
)sequential/lstm/lstm_cell/clip_by_value_1Maximum5sequential/lstm/lstm_cell/clip_by_value_1/Minimum:z:04sequential/lstm/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2+
)sequential/lstm/lstm_cell/clip_by_value_1Í
sequential/lstm/lstm_cell/mul_2Mul-sequential/lstm/lstm_cell/clip_by_value_1:z:0 sequential/lstm/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2!
sequential/lstm/lstm_cell/mul_2Ì
*sequential/lstm/lstm_cell/ReadVariableOp_2ReadVariableOp1sequential_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
È *
dtype02,
*sequential/lstm/lstm_cell/ReadVariableOp_2³
/sequential/lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"      21
/sequential/lstm/lstm_cell/strided_slice_2/stack·
1sequential/lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  23
1sequential/lstm/lstm_cell/strided_slice_2/stack_1·
1sequential/lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1sequential/lstm/lstm_cell/strided_slice_2/stack_2¦
)sequential/lstm/lstm_cell/strided_slice_2StridedSlice2sequential/lstm/lstm_cell/ReadVariableOp_2:value:08sequential/lstm/lstm_cell/strided_slice_2/stack:output:0:sequential/lstm/lstm_cell/strided_slice_2/stack_1:output:0:sequential/lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2+
)sequential/lstm/lstm_cell/strided_slice_2Ù
"sequential/lstm/lstm_cell/MatMul_6MatMulsequential/lstm/zeros:output:02sequential/lstm/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2$
"sequential/lstm/lstm_cell/MatMul_6Ú
sequential/lstm/lstm_cell/add_4AddV2,sequential/lstm/lstm_cell/BiasAdd_2:output:0,sequential/lstm/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2!
sequential/lstm/lstm_cell/add_4 
sequential/lstm/lstm_cell/TanhTanh#sequential/lstm/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2 
sequential/lstm/lstm_cell/TanhÍ
sequential/lstm/lstm_cell/mul_3Mul+sequential/lstm/lstm_cell/clip_by_value:z:0"sequential/lstm/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2!
sequential/lstm/lstm_cell/mul_3È
sequential/lstm/lstm_cell/add_5AddV2#sequential/lstm/lstm_cell/mul_2:z:0#sequential/lstm/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2!
sequential/lstm/lstm_cell/add_5Ì
*sequential/lstm/lstm_cell/ReadVariableOp_3ReadVariableOp1sequential_lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
È *
dtype02,
*sequential/lstm/lstm_cell/ReadVariableOp_3³
/sequential/lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  21
/sequential/lstm/lstm_cell/strided_slice_3/stack·
1sequential/lstm/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        23
1sequential/lstm/lstm_cell/strided_slice_3/stack_1·
1sequential/lstm/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      23
1sequential/lstm/lstm_cell/strided_slice_3/stack_2¦
)sequential/lstm/lstm_cell/strided_slice_3StridedSlice2sequential/lstm/lstm_cell/ReadVariableOp_3:value:08sequential/lstm/lstm_cell/strided_slice_3/stack:output:0:sequential/lstm/lstm_cell/strided_slice_3/stack_1:output:0:sequential/lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2+
)sequential/lstm/lstm_cell/strided_slice_3Ù
"sequential/lstm/lstm_cell/MatMul_7MatMulsequential/lstm/zeros:output:02sequential/lstm/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2$
"sequential/lstm/lstm_cell/MatMul_7Ú
sequential/lstm/lstm_cell/add_6AddV2,sequential/lstm/lstm_cell/BiasAdd_3:output:0,sequential/lstm/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2!
sequential/lstm/lstm_cell/add_6
!sequential/lstm/lstm_cell/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2#
!sequential/lstm/lstm_cell/Const_6
!sequential/lstm/lstm_cell/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!sequential/lstm/lstm_cell/Const_7Í
sequential/lstm/lstm_cell/Mul_4Mul#sequential/lstm/lstm_cell/add_6:z:0*sequential/lstm/lstm_cell/Const_6:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2!
sequential/lstm/lstm_cell/Mul_4Í
sequential/lstm/lstm_cell/Add_7Add#sequential/lstm/lstm_cell/Mul_4:z:0*sequential/lstm/lstm_cell/Const_7:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2!
sequential/lstm/lstm_cell/Add_7¯
3sequential/lstm/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?25
3sequential/lstm/lstm_cell/clip_by_value_2/Minimum/y
1sequential/lstm/lstm_cell/clip_by_value_2/MinimumMinimum#sequential/lstm/lstm_cell/Add_7:z:0<sequential/lstm/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ23
1sequential/lstm/lstm_cell/clip_by_value_2/Minimum
+sequential/lstm/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2-
+sequential/lstm/lstm_cell/clip_by_value_2/y
)sequential/lstm/lstm_cell/clip_by_value_2Maximum5sequential/lstm/lstm_cell/clip_by_value_2/Minimum:z:04sequential/lstm/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2+
)sequential/lstm/lstm_cell/clip_by_value_2¤
 sequential/lstm/lstm_cell/Tanh_1Tanh#sequential/lstm/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2"
 sequential/lstm/lstm_cell/Tanh_1Ñ
sequential/lstm/lstm_cell/mul_5Mul-sequential/lstm/lstm_cell/clip_by_value_2:z:0$sequential/lstm/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2!
sequential/lstm/lstm_cell/mul_5¯
-sequential/lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   2/
-sequential/lstm/TensorArrayV2_1/element_shapeø
sequential/lstm/TensorArrayV2_1TensorListReserve6sequential/lstm/TensorArrayV2_1/element_shape:output:0(sequential/lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02!
sequential/lstm/TensorArrayV2_1n
sequential/lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
sequential/lstm/time
(sequential/lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2*
(sequential/lstm/while/maximum_iterations
"sequential/lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2$
"sequential/lstm/while/loop_counterË
sequential/lstm/whileWhile+sequential/lstm/while/loop_counter:output:01sequential/lstm/while/maximum_iterations:output:0sequential/lstm/time:output:0(sequential/lstm/TensorArrayV2_1:handle:0sequential/lstm/zeros:output:0 sequential/lstm/zeros_1:output:0(sequential/lstm/strided_slice_1:output:0Gsequential/lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:07sequential_lstm_lstm_cell_split_readvariableop_resource9sequential_lstm_lstm_cell_split_1_readvariableop_resource1sequential_lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *%
_read_only_resource_inputs
	
*+
body#R!
sequential_lstm_while_body_9913*+
cond#R!
sequential_lstm_while_cond_9912*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *
parallel_iterations 2
sequential/lstm/whileÕ
@sequential/lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   2B
@sequential/lstm/TensorArrayV2Stack/TensorListStack/element_shape²
2sequential/lstm/TensorArrayV2Stack/TensorListStackTensorListStacksequential/lstm/while:output:3Isequential/lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ*
element_dtype024
2sequential/lstm/TensorArrayV2Stack/TensorListStack¡
%sequential/lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2'
%sequential/lstm/strided_slice_3/stack
'sequential/lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2)
'sequential/lstm/strided_slice_3/stack_1
'sequential/lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'sequential/lstm/strided_slice_3/stack_2û
sequential/lstm/strided_slice_3StridedSlice;sequential/lstm/TensorArrayV2Stack/TensorListStack:tensor:0.sequential/lstm/strided_slice_3/stack:output:00sequential/lstm/strided_slice_3/stack_1:output:00sequential/lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask2!
sequential/lstm/strided_slice_3
 sequential/lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2"
 sequential/lstm/transpose_1/permï
sequential/lstm/transpose_1	Transpose;sequential/lstm/TensorArrayV2Stack/TensorListStack:tensor:0)sequential/lstm/transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
sequential/lstm/transpose_1
sequential/lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
sequential/lstm/runtime§
sequential/dropout/IdentityIdentitysequential/lstm/transpose_1:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
sequential/dropout/IdentityÊ
)sequential/dense/Tensordot/ReadVariableOpReadVariableOp2sequential_dense_tensordot_readvariableop_resource*
_output_shapes
:	È*
dtype02+
)sequential/dense/Tensordot/ReadVariableOp
sequential/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2!
sequential/dense/Tensordot/axes
sequential/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2!
sequential/dense/Tensordot/free
 sequential/dense/Tensordot/ShapeShape$sequential/dropout/Identity:output:0*
T0*
_output_shapes
:2"
 sequential/dense/Tensordot/Shape
(sequential/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/dense/Tensordot/GatherV2/axis¦
#sequential/dense/Tensordot/GatherV2GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/free:output:01sequential/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2%
#sequential/dense/Tensordot/GatherV2
*sequential/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential/dense/Tensordot/GatherV2_1/axis¬
%sequential/dense/Tensordot/GatherV2_1GatherV2)sequential/dense/Tensordot/Shape:output:0(sequential/dense/Tensordot/axes:output:03sequential/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2'
%sequential/dense/Tensordot/GatherV2_1
 sequential/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 sequential/dense/Tensordot/ConstÄ
sequential/dense/Tensordot/ProdProd,sequential/dense/Tensordot/GatherV2:output:0)sequential/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2!
sequential/dense/Tensordot/Prod
"sequential/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2$
"sequential/dense/Tensordot/Const_1Ì
!sequential/dense/Tensordot/Prod_1Prod.sequential/dense/Tensordot/GatherV2_1:output:0+sequential/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2#
!sequential/dense/Tensordot/Prod_1
&sequential/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2(
&sequential/dense/Tensordot/concat/axis
!sequential/dense/Tensordot/concatConcatV2(sequential/dense/Tensordot/free:output:0(sequential/dense/Tensordot/axes:output:0/sequential/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2#
!sequential/dense/Tensordot/concatÐ
 sequential/dense/Tensordot/stackPack(sequential/dense/Tensordot/Prod:output:0*sequential/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2"
 sequential/dense/Tensordot/stackë
$sequential/dense/Tensordot/transpose	Transpose$sequential/dropout/Identity:output:0*sequential/dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2&
$sequential/dense/Tensordot/transposeã
"sequential/dense/Tensordot/ReshapeReshape(sequential/dense/Tensordot/transpose:y:0)sequential/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2$
"sequential/dense/Tensordot/Reshapeâ
!sequential/dense/Tensordot/MatMulMatMul+sequential/dense/Tensordot/Reshape:output:01sequential/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2#
!sequential/dense/Tensordot/MatMul
"sequential/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"sequential/dense/Tensordot/Const_2
(sequential/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/dense/Tensordot/concat_1/axis
#sequential/dense/Tensordot/concat_1ConcatV2,sequential/dense/Tensordot/GatherV2:output:0+sequential/dense/Tensordot/Const_2:output:01sequential/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2%
#sequential/dense/Tensordot/concat_1Ý
sequential/dense/TensordotReshape+sequential/dense/Tensordot/MatMul:product:0,sequential/dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
sequential/dense/Tensordot¿
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOpÔ
sequential/dense/BiasAddBiasAdd#sequential/dense/Tensordot:output:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
sequential/dense/BiasAdd¬
sequential/dropout_1/IdentityIdentity!sequential/dense/BiasAdd:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
sequential/dropout_1/IdentityÏ
+sequential/dense_1/Tensordot/ReadVariableOpReadVariableOp4sequential_dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02-
+sequential/dense_1/Tensordot/ReadVariableOp
!sequential/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2#
!sequential/dense_1/Tensordot/axes
!sequential/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2#
!sequential/dense_1/Tensordot/free
"sequential/dense_1/Tensordot/ShapeShape&sequential/dropout_1/Identity:output:0*
T0*
_output_shapes
:2$
"sequential/dense_1/Tensordot/Shape
*sequential/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential/dense_1/Tensordot/GatherV2/axis°
%sequential/dense_1/Tensordot/GatherV2GatherV2+sequential/dense_1/Tensordot/Shape:output:0*sequential/dense_1/Tensordot/free:output:03sequential/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2'
%sequential/dense_1/Tensordot/GatherV2
,sequential/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2.
,sequential/dense_1/Tensordot/GatherV2_1/axis¶
'sequential/dense_1/Tensordot/GatherV2_1GatherV2+sequential/dense_1/Tensordot/Shape:output:0*sequential/dense_1/Tensordot/axes:output:05sequential/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2)
'sequential/dense_1/Tensordot/GatherV2_1
"sequential/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"sequential/dense_1/Tensordot/ConstÌ
!sequential/dense_1/Tensordot/ProdProd.sequential/dense_1/Tensordot/GatherV2:output:0+sequential/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2#
!sequential/dense_1/Tensordot/Prod
$sequential/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$sequential/dense_1/Tensordot/Const_1Ô
#sequential/dense_1/Tensordot/Prod_1Prod0sequential/dense_1/Tensordot/GatherV2_1:output:0-sequential/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2%
#sequential/dense_1/Tensordot/Prod_1
(sequential/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2*
(sequential/dense_1/Tensordot/concat/axis
#sequential/dense_1/Tensordot/concatConcatV2*sequential/dense_1/Tensordot/free:output:0*sequential/dense_1/Tensordot/axes:output:01sequential/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2%
#sequential/dense_1/Tensordot/concatØ
"sequential/dense_1/Tensordot/stackPack*sequential/dense_1/Tensordot/Prod:output:0,sequential/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2$
"sequential/dense_1/Tensordot/stackò
&sequential/dense_1/Tensordot/transpose	Transpose&sequential/dropout_1/Identity:output:0,sequential/dense_1/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2(
&sequential/dense_1/Tensordot/transposeë
$sequential/dense_1/Tensordot/ReshapeReshape*sequential/dense_1/Tensordot/transpose:y:0+sequential/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2&
$sequential/dense_1/Tensordot/Reshapeê
#sequential/dense_1/Tensordot/MatMulMatMul-sequential/dense_1/Tensordot/Reshape:output:03sequential/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2%
#sequential/dense_1/Tensordot/MatMul
$sequential/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$sequential/dense_1/Tensordot/Const_2
*sequential/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2,
*sequential/dense_1/Tensordot/concat_1/axis
%sequential/dense_1/Tensordot/concat_1ConcatV2.sequential/dense_1/Tensordot/GatherV2:output:0-sequential/dense_1/Tensordot/Const_2:output:03sequential/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2'
%sequential/dense_1/Tensordot/concat_1å
sequential/dense_1/TensordotReshape-sequential/dense_1/Tensordot/MatMul:product:0.sequential/dense_1/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
sequential/dense_1/TensordotÅ
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOp2sequential_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)sequential/dense_1/BiasAdd/ReadVariableOpÜ
sequential/dense_1/BiasAddBiasAdd%sequential/dense_1/Tensordot:output:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
sequential/dense_1/BiasAdd
IdentityIdentity#sequential/dense_1/BiasAdd:output:0^sequential/lstm/while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::::2.
sequential/lstm/whilesequential/lstm/while:a ]
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
lstm_input
º$
õ
while_body_10758
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_10782_0
while_lstm_cell_10784_0
while_lstm_cell_10786_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_10782
while_lstm_cell_10784
while_lstm_cell_10786¢'while/lstm_cell/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÎ
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_10782_0while_lstm_cell_10784_0while_lstm_cell_10786_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_103322)
'while/lstm_cell/StatefulPartitionedCallô
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2·
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3¿
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1(^while/lstm_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/Identity_4¿
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2(^while/lstm_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"0
while_lstm_cell_10782while_lstm_cell_10782_0"0
while_lstm_cell_10784while_lstm_cell_10784_0"0
while_lstm_cell_10786while_lstm_cell_10786_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : :::2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
: 
±

$__inference_lstm_layer_call_fn_12944
inputs_0
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_106952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
´D
É
?__inference_lstm_layer_call_and_return_conditional_losses_10695

inputs
lstm_cell_10613
lstm_cell_10615
lstm_cell_10617
identity¢!lstm_cell/StatefulPartitionedCall¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_10613lstm_cell_10615lstm_cell_10617*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_102412#
!lstm_cell/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_10613lstm_cell_10615lstm_cell_10617*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_10626*
condR
while_cond_10625*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitytranspose_1:y:0"^lstm_cell/StatefulPartitionedCall^while*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò
b
)__inference_dropout_1_layer_call_fn_13603

inputs
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_114952
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

|
'__inference_dense_1_layer_call_fn_13647

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallÿ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_115432
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±

$__inference_lstm_layer_call_fn_12955
inputs_0
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_108272
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::22
StatefulPartitionedCallStatefulPartitionedCall:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Ð
Ó
while_body_10965
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
/while_lstm_cell_split_readvariableop_resource_05
1while_lstm_cell_split_1_readvariableop_resource_0-
)while_lstm_cell_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
-while_lstm_cell_split_readvariableop_resource3
/while_lstm_cell_split_1_readvariableop_resource+
'while_lstm_cell_readvariableop_resourceÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim¾
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
 *
dtype02&
$while/lstm_cell/split/ReadVariableOpï
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È:
È:
È:
È*
	num_split2
while/lstm_cell/split¿
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMulÃ
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_1Ã
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_2Ã
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_3t
while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const_1
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!while/lstm_cell/split_1/split_dim¿
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
: *
dtype02(
&while/lstm_cell/split_1/ReadVariableOpã
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:È:È:È:È*
	num_split2
while/lstm_cell/split_1´
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/BiasAddº
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/BiasAdd_1º
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/BiasAdd_2º
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/BiasAdd_3¬
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02 
while/lstm_cell/ReadVariableOp
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#while/lstm_cell/strided_slice/stack
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    È   2'
%while/lstm_cell/strided_slice/stack_1
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice/stack_2Þ
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
while/lstm_cell/strided_slice®
while/lstm_cell/MatMul_4MatMulwhile_placeholder_2&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_4¬
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/addw
while/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
while/lstm_cell/Const_2w
while/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/Const_3
while/lstm_cell/MulMulwhile/lstm_cell/add:z:0 while/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Mul£
while/lstm_cell/Add_1Addwhile/lstm_cell/Mul:z:0 while/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Add_1
'while/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'while/lstm_cell/clip_by_value/Minimum/yÙ
%while/lstm_cell/clip_by_value/MinimumMinimumwhile/lstm_cell/Add_1:z:00while/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2'
%while/lstm_cell/clip_by_value/Minimum
while/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
while/lstm_cell/clip_by_value/yÑ
while/lstm_cell/clip_by_valueMaximum)while/lstm_cell/clip_by_value/Minimum:z:0(while/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/clip_by_value°
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02"
 while/lstm_cell/ReadVariableOp_1
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    È   2'
%while/lstm_cell/strided_slice_1/stack£
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_1£
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_2ê
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_1°
while/lstm_cell/MatMul_5MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_5²
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/add_2w
while/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
while/lstm_cell/Const_4w
while/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/Const_5¥
while/lstm_cell/Mul_1Mulwhile/lstm_cell/add_2:z:0 while/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Mul_1¥
while/lstm_cell/Add_3Addwhile/lstm_cell/Mul_1:z:0 while/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Add_3
)while/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)while/lstm_cell/clip_by_value_1/Minimum/yß
'while/lstm_cell/clip_by_value_1/MinimumMinimumwhile/lstm_cell/Add_3:z:02while/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2)
'while/lstm_cell/clip_by_value_1/Minimum
!while/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!while/lstm_cell/clip_by_value_1/yÙ
while/lstm_cell/clip_by_value_1Maximum+while/lstm_cell/clip_by_value_1/Minimum:z:0*while/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2!
while/lstm_cell/clip_by_value_1¢
while/lstm_cell/mul_2Mul#while/lstm_cell/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/mul_2°
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02"
 while/lstm_cell/ReadVariableOp_2
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice_2/stack£
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2)
'while/lstm_cell/strided_slice_2/stack_1£
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_2/stack_2ê
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_2°
while/lstm_cell/MatMul_6MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_6²
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/add_4
while/lstm_cell/TanhTanhwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Tanh¥
while/lstm_cell/mul_3Mul!while/lstm_cell/clip_by_value:z:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/mul_3 
while/lstm_cell/add_5AddV2while/lstm_cell/mul_2:z:0while/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/add_5°
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02"
 while/lstm_cell/ReadVariableOp_3
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2'
%while/lstm_cell/strided_slice_3/stack£
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell/strided_slice_3/stack_1£
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_3/stack_2ê
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_3°
while/lstm_cell/MatMul_7MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_7²
while/lstm_cell/add_6AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/add_6w
while/lstm_cell/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
while/lstm_cell/Const_6w
while/lstm_cell/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/Const_7¥
while/lstm_cell/Mul_4Mulwhile/lstm_cell/add_6:z:0 while/lstm_cell/Const_6:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Mul_4¥
while/lstm_cell/Add_7Addwhile/lstm_cell/Mul_4:z:0 while/lstm_cell/Const_7:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Add_7
)while/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)while/lstm_cell/clip_by_value_2/Minimum/yß
'while/lstm_cell/clip_by_value_2/MinimumMinimumwhile/lstm_cell/Add_7:z:02while/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2)
'while/lstm_cell/clip_by_value_2/Minimum
!while/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!while/lstm_cell/clip_by_value_2/yÙ
while/lstm_cell/clip_by_value_2Maximum+while/lstm_cell/clip_by_value_2/Minimum:z:0*while/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2!
while/lstm_cell/clip_by_value_2
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Tanh_1©
while/lstm_cell/mul_5Mul#while/lstm_cell/clip_by_value_2:z:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/mul_5Ý
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3~
while/Identity_4Identitywhile/lstm_cell/mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/Identity_4~
while/Identity_5Identitywhile/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
: 
Á
Î
*__inference_sequential_layer_call_fn_11668

lstm_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity¢StatefulPartitionedCallÇ
StatefulPartitionedCallStatefulPartitionedCall
lstm_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_116512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
lstm_input
¿
É
)__inference_lstm_cell_layer_call_fn_13846

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2¢StatefulPartitionedCallÂ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_102412
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*[
_input_shapesJ
H:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
"
_user_specified_name
states/1
ª
¾
while_cond_10757
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_10757___redundant_placeholder03
/while_while_cond_10757___redundant_placeholder13
/while_while_cond_10757___redundant_placeholder23
/while_while_cond_10757___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
:
Ð
Ó
while_body_11234
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
/while_lstm_cell_split_readvariableop_resource_05
1while_lstm_cell_split_1_readvariableop_resource_0-
)while_lstm_cell_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
-while_lstm_cell_split_readvariableop_resource3
/while_lstm_cell_split_1_readvariableop_resource+
'while_lstm_cell_readvariableop_resourceÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim¾
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
 *
dtype02&
$while/lstm_cell/split/ReadVariableOpï
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È:
È:
È:
È*
	num_split2
while/lstm_cell/split¿
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMulÃ
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_1Ã
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_2Ã
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_3t
while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const_1
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!while/lstm_cell/split_1/split_dim¿
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
: *
dtype02(
&while/lstm_cell/split_1/ReadVariableOpã
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:È:È:È:È*
	num_split2
while/lstm_cell/split_1´
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/BiasAddº
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/BiasAdd_1º
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/BiasAdd_2º
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/BiasAdd_3¬
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02 
while/lstm_cell/ReadVariableOp
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#while/lstm_cell/strided_slice/stack
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    È   2'
%while/lstm_cell/strided_slice/stack_1
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice/stack_2Þ
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
while/lstm_cell/strided_slice®
while/lstm_cell/MatMul_4MatMulwhile_placeholder_2&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_4¬
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/addw
while/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
while/lstm_cell/Const_2w
while/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/Const_3
while/lstm_cell/MulMulwhile/lstm_cell/add:z:0 while/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Mul£
while/lstm_cell/Add_1Addwhile/lstm_cell/Mul:z:0 while/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Add_1
'while/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'while/lstm_cell/clip_by_value/Minimum/yÙ
%while/lstm_cell/clip_by_value/MinimumMinimumwhile/lstm_cell/Add_1:z:00while/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2'
%while/lstm_cell/clip_by_value/Minimum
while/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
while/lstm_cell/clip_by_value/yÑ
while/lstm_cell/clip_by_valueMaximum)while/lstm_cell/clip_by_value/Minimum:z:0(while/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/clip_by_value°
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02"
 while/lstm_cell/ReadVariableOp_1
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    È   2'
%while/lstm_cell/strided_slice_1/stack£
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_1£
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_2ê
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_1°
while/lstm_cell/MatMul_5MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_5²
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/add_2w
while/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
while/lstm_cell/Const_4w
while/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/Const_5¥
while/lstm_cell/Mul_1Mulwhile/lstm_cell/add_2:z:0 while/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Mul_1¥
while/lstm_cell/Add_3Addwhile/lstm_cell/Mul_1:z:0 while/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Add_3
)while/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)while/lstm_cell/clip_by_value_1/Minimum/yß
'while/lstm_cell/clip_by_value_1/MinimumMinimumwhile/lstm_cell/Add_3:z:02while/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2)
'while/lstm_cell/clip_by_value_1/Minimum
!while/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!while/lstm_cell/clip_by_value_1/yÙ
while/lstm_cell/clip_by_value_1Maximum+while/lstm_cell/clip_by_value_1/Minimum:z:0*while/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2!
while/lstm_cell/clip_by_value_1¢
while/lstm_cell/mul_2Mul#while/lstm_cell/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/mul_2°
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02"
 while/lstm_cell/ReadVariableOp_2
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice_2/stack£
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2)
'while/lstm_cell/strided_slice_2/stack_1£
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_2/stack_2ê
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_2°
while/lstm_cell/MatMul_6MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_6²
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/add_4
while/lstm_cell/TanhTanhwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Tanh¥
while/lstm_cell/mul_3Mul!while/lstm_cell/clip_by_value:z:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/mul_3 
while/lstm_cell/add_5AddV2while/lstm_cell/mul_2:z:0while/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/add_5°
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02"
 while/lstm_cell/ReadVariableOp_3
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2'
%while/lstm_cell/strided_slice_3/stack£
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell/strided_slice_3/stack_1£
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_3/stack_2ê
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_3°
while/lstm_cell/MatMul_7MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_7²
while/lstm_cell/add_6AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/add_6w
while/lstm_cell/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
while/lstm_cell/Const_6w
while/lstm_cell/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/Const_7¥
while/lstm_cell/Mul_4Mulwhile/lstm_cell/add_6:z:0 while/lstm_cell/Const_6:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Mul_4¥
while/lstm_cell/Add_7Addwhile/lstm_cell/Mul_4:z:0 while/lstm_cell/Const_7:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Add_7
)while/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)while/lstm_cell/clip_by_value_2/Minimum/yß
'while/lstm_cell/clip_by_value_2/MinimumMinimumwhile/lstm_cell/Add_7:z:02while/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2)
'while/lstm_cell/clip_by_value_2/Minimum
!while/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!while/lstm_cell/clip_by_value_2/yÙ
while/lstm_cell/clip_by_value_2Maximum+while/lstm_cell/clip_by_value_2/Minimum:z:0*while/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2!
while/lstm_cell/clip_by_value_2
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Tanh_1©
while/lstm_cell/mul_5Mul#while/lstm_cell/clip_by_value_2:z:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/mul_5Ý
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3~
while/Identity_4Identitywhile/lstm_cell/mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/Identity_4~
while/Identity_5Identitywhile/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
: 
«

$__inference_lstm_layer_call_fn_13504

inputs
unknown
	unknown_0
	unknown_1
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_111082
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ôî
·
E__inference_sequential_layer_call_and_return_conditional_losses_12357

inputs0
,lstm_lstm_cell_split_readvariableop_resource2
.lstm_lstm_cell_split_1_readvariableop_resource*
&lstm_lstm_cell_readvariableop_resource+
'dense_tensordot_readvariableop_resource)
%dense_biasadd_readvariableop_resource-
)dense_1_tensordot_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity¢
lstm/whileN

lstm/ShapeShapeinputs*
T0*
_output_shapes
:2

lstm/Shape~
lstm/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice/stack
lstm/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_1
lstm/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice/stack_2
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_sliceg
lstm/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
lstm/zeros/mul/y
lstm/zeros/mulMullstm/strided_slice:output:0lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/muli
lstm/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm/zeros/Less/y{
lstm/zeros/LessLesslstm/zeros/mul:z:0lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros/Lessm
lstm/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
lstm/zeros/packed/1
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros/packedi
lstm/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros/Const

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

lstm/zerosk
lstm/zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
lstm/zeros_1/mul/y
lstm/zeros_1/mulMullstm/strided_slice:output:0lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/mulm
lstm/zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
lstm/zeros_1/Less/y
lstm/zeros_1/LessLesslstm/zeros_1/mul:z:0lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
lstm/zeros_1/Lessq
lstm/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
lstm/zeros_1/packed/1
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
lstm/zeros_1/packedm
lstm/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/zeros_1/Const
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/zeros_1
lstm/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose/perm
lstm/transpose	Transposeinputslstm/transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
lstm/transpose^
lstm/Shape_1Shapelstm/transpose:y:0*
T0*
_output_shapes
:2
lstm/Shape_1
lstm/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_1/stack
lstm/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_1
lstm/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_1/stack_2
lstm/strided_slice_1StridedSlicelstm/Shape_1:output:0#lstm/strided_slice_1/stack:output:0%lstm/strided_slice_1/stack_1:output:0%lstm/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
lstm/strided_slice_1
 lstm/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2"
 lstm/TensorArrayV2/element_shapeÆ
lstm/TensorArrayV2TensorListReserve)lstm/TensorArrayV2/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2É
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2<
:lstm/TensorArrayUnstack/TensorListFromTensor/element_shape
,lstm/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm/transpose:y:0Clstm/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02.
,lstm/TensorArrayUnstack/TensorListFromTensor
lstm/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_2/stack
lstm/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_1
lstm/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_2/stack_2
lstm/strided_slice_2StridedSlicelstm/transpose:y:0#lstm/strided_slice_2/stack:output:0%lstm/strided_slice_2/stack_1:output:0%lstm/strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
lstm/strided_slice_2n
lstm/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/lstm_cell/Const
lstm/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2 
lstm/lstm_cell/split/split_dim¹
#lstm/lstm_cell/split/ReadVariableOpReadVariableOp,lstm_lstm_cell_split_readvariableop_resource* 
_output_shapes
:
 *
dtype02%
#lstm/lstm_cell/split/ReadVariableOpë
lstm/lstm_cell/splitSplit'lstm/lstm_cell/split/split_dim:output:0+lstm/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È:
È:
È:
È*
	num_split2
lstm/lstm_cell/split©
lstm/lstm_cell/MatMulMatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/MatMul­
lstm/lstm_cell/MatMul_1MatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/MatMul_1­
lstm/lstm_cell/MatMul_2MatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/MatMul_2­
lstm/lstm_cell/MatMul_3MatMullstm/strided_slice_2:output:0lstm/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/MatMul_3r
lstm/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm/lstm_cell/Const_1
 lstm/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2"
 lstm/lstm_cell/split_1/split_dimº
%lstm/lstm_cell/split_1/ReadVariableOpReadVariableOp.lstm_lstm_cell_split_1_readvariableop_resource*
_output_shapes	
: *
dtype02'
%lstm/lstm_cell/split_1/ReadVariableOpß
lstm/lstm_cell/split_1Split)lstm/lstm_cell/split_1/split_dim:output:0-lstm/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:È:È:È:È*
	num_split2
lstm/lstm_cell/split_1°
lstm/lstm_cell/BiasAddBiasAddlstm/lstm_cell/MatMul:product:0lstm/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/BiasAdd¶
lstm/lstm_cell/BiasAdd_1BiasAdd!lstm/lstm_cell/MatMul_1:product:0lstm/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/BiasAdd_1¶
lstm/lstm_cell/BiasAdd_2BiasAdd!lstm/lstm_cell/MatMul_2:product:0lstm/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/BiasAdd_2¶
lstm/lstm_cell/BiasAdd_3BiasAdd!lstm/lstm_cell/MatMul_3:product:0lstm/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/BiasAdd_3§
lstm/lstm_cell/ReadVariableOpReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
È *
dtype02
lstm/lstm_cell/ReadVariableOp
"lstm/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2$
"lstm/lstm_cell/strided_slice/stack
$lstm/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    È   2&
$lstm/lstm_cell/strided_slice/stack_1
$lstm/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm/lstm_cell/strided_slice/stack_2Ø
lstm/lstm_cell/strided_sliceStridedSlice%lstm/lstm_cell/ReadVariableOp:value:0+lstm/lstm_cell/strided_slice/stack:output:0-lstm/lstm_cell/strided_slice/stack_1:output:0-lstm/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
lstm/lstm_cell/strided_slice«
lstm/lstm_cell/MatMul_4MatMullstm/zeros:output:0%lstm/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/MatMul_4¨
lstm/lstm_cell/addAddV2lstm/lstm_cell/BiasAdd:output:0!lstm/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/addu
lstm/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
lstm/lstm_cell/Const_2u
lstm/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm/lstm_cell/Const_3
lstm/lstm_cell/MulMullstm/lstm_cell/add:z:0lstm/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/Mul
lstm/lstm_cell/Add_1Addlstm/lstm_cell/Mul:z:0lstm/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/Add_1
&lstm/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2(
&lstm/lstm_cell/clip_by_value/Minimum/yÕ
$lstm/lstm_cell/clip_by_value/MinimumMinimumlstm/lstm_cell/Add_1:z:0/lstm/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2&
$lstm/lstm_cell/clip_by_value/Minimum
lstm/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2 
lstm/lstm_cell/clip_by_value/yÍ
lstm/lstm_cell/clip_by_valueMaximum(lstm/lstm_cell/clip_by_value/Minimum:z:0'lstm/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/clip_by_value«
lstm/lstm_cell/ReadVariableOp_1ReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
È *
dtype02!
lstm/lstm_cell/ReadVariableOp_1
$lstm/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    È   2&
$lstm/lstm_cell/strided_slice_1/stack¡
&lstm/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2(
&lstm/lstm_cell/strided_slice_1/stack_1¡
&lstm/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&lstm/lstm_cell/strided_slice_1/stack_2ä
lstm/lstm_cell/strided_slice_1StridedSlice'lstm/lstm_cell/ReadVariableOp_1:value:0-lstm/lstm_cell/strided_slice_1/stack:output:0/lstm/lstm_cell/strided_slice_1/stack_1:output:0/lstm/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2 
lstm/lstm_cell/strided_slice_1­
lstm/lstm_cell/MatMul_5MatMullstm/zeros:output:0'lstm/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/MatMul_5®
lstm/lstm_cell/add_2AddV2!lstm/lstm_cell/BiasAdd_1:output:0!lstm/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/add_2u
lstm/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
lstm/lstm_cell/Const_4u
lstm/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm/lstm_cell/Const_5¡
lstm/lstm_cell/Mul_1Mullstm/lstm_cell/add_2:z:0lstm/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/Mul_1¡
lstm/lstm_cell/Add_3Addlstm/lstm_cell/Mul_1:z:0lstm/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/Add_3
(lstm/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2*
(lstm/lstm_cell/clip_by_value_1/Minimum/yÛ
&lstm/lstm_cell/clip_by_value_1/MinimumMinimumlstm/lstm_cell/Add_3:z:01lstm/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2(
&lstm/lstm_cell/clip_by_value_1/Minimum
 lstm/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 lstm/lstm_cell/clip_by_value_1/yÕ
lstm/lstm_cell/clip_by_value_1Maximum*lstm/lstm_cell/clip_by_value_1/Minimum:z:0)lstm/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2 
lstm/lstm_cell/clip_by_value_1¡
lstm/lstm_cell/mul_2Mul"lstm/lstm_cell/clip_by_value_1:z:0lstm/zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/mul_2«
lstm/lstm_cell/ReadVariableOp_2ReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
È *
dtype02!
lstm/lstm_cell/ReadVariableOp_2
$lstm/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"      2&
$lstm/lstm_cell/strided_slice_2/stack¡
&lstm/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2(
&lstm/lstm_cell/strided_slice_2/stack_1¡
&lstm/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&lstm/lstm_cell/strided_slice_2/stack_2ä
lstm/lstm_cell/strided_slice_2StridedSlice'lstm/lstm_cell/ReadVariableOp_2:value:0-lstm/lstm_cell/strided_slice_2/stack:output:0/lstm/lstm_cell/strided_slice_2/stack_1:output:0/lstm/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2 
lstm/lstm_cell/strided_slice_2­
lstm/lstm_cell/MatMul_6MatMullstm/zeros:output:0'lstm/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/MatMul_6®
lstm/lstm_cell/add_4AddV2!lstm/lstm_cell/BiasAdd_2:output:0!lstm/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/add_4
lstm/lstm_cell/TanhTanhlstm/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/Tanh¡
lstm/lstm_cell/mul_3Mul lstm/lstm_cell/clip_by_value:z:0lstm/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/mul_3
lstm/lstm_cell/add_5AddV2lstm/lstm_cell/mul_2:z:0lstm/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/add_5«
lstm/lstm_cell/ReadVariableOp_3ReadVariableOp&lstm_lstm_cell_readvariableop_resource* 
_output_shapes
:
È *
dtype02!
lstm/lstm_cell/ReadVariableOp_3
$lstm/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2&
$lstm/lstm_cell/strided_slice_3/stack¡
&lstm/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2(
&lstm/lstm_cell/strided_slice_3/stack_1¡
&lstm/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2(
&lstm/lstm_cell/strided_slice_3/stack_2ä
lstm/lstm_cell/strided_slice_3StridedSlice'lstm/lstm_cell/ReadVariableOp_3:value:0-lstm/lstm_cell/strided_slice_3/stack:output:0/lstm/lstm_cell/strided_slice_3/stack_1:output:0/lstm/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2 
lstm/lstm_cell/strided_slice_3­
lstm/lstm_cell/MatMul_7MatMullstm/zeros:output:0'lstm/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/MatMul_7®
lstm/lstm_cell/add_6AddV2!lstm/lstm_cell/BiasAdd_3:output:0!lstm/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/add_6u
lstm/lstm_cell/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
lstm/lstm_cell/Const_6u
lstm/lstm_cell/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm/lstm_cell/Const_7¡
lstm/lstm_cell/Mul_4Mullstm/lstm_cell/add_6:z:0lstm/lstm_cell/Const_6:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/Mul_4¡
lstm/lstm_cell/Add_7Addlstm/lstm_cell/Mul_4:z:0lstm/lstm_cell/Const_7:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/Add_7
(lstm/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2*
(lstm/lstm_cell/clip_by_value_2/Minimum/yÛ
&lstm/lstm_cell/clip_by_value_2/MinimumMinimumlstm/lstm_cell/Add_7:z:01lstm/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2(
&lstm/lstm_cell/clip_by_value_2/Minimum
 lstm/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2"
 lstm/lstm_cell/clip_by_value_2/yÕ
lstm/lstm_cell/clip_by_value_2Maximum*lstm/lstm_cell/clip_by_value_2/Minimum:z:0)lstm/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2 
lstm/lstm_cell/clip_by_value_2
lstm/lstm_cell/Tanh_1Tanhlstm/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/Tanh_1¥
lstm/lstm_cell/mul_5Mul"lstm/lstm_cell/clip_by_value_2:z:0lstm/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/lstm_cell/mul_5
"lstm/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   2$
"lstm/TensorArrayV2_1/element_shapeÌ
lstm/TensorArrayV2_1TensorListReserve+lstm/TensorArrayV2_1/element_shape:output:0lstm/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
lstm/TensorArrayV2_1X
	lstm/timeConst*
_output_shapes
: *
dtype0*
value	B : 2
	lstm/time
lstm/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
lstm/while/maximum_iterationst
lstm/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm/while/loop_counter¨

lstm/whileWhile lstm/while/loop_counter:output:0&lstm/while/maximum_iterations:output:0lstm/time:output:0lstm/TensorArrayV2_1:handle:0lstm/zeros:output:0lstm/zeros_1:output:0lstm/strided_slice_1:output:0<lstm/TensorArrayUnstack/TensorListFromTensor:output_handle:0,lstm_lstm_cell_split_readvariableop_resource.lstm_lstm_cell_split_1_readvariableop_resource&lstm_lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *%
_read_only_resource_inputs
	
*!
bodyR
lstm_while_body_12160*!
condR
lstm_while_cond_12159*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *
parallel_iterations 2

lstm/while¿
5lstm/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   27
5lstm/TensorArrayV2Stack/TensorListStack/element_shape
'lstm/TensorArrayV2Stack/TensorListStackTensorListStacklstm/while:output:3>lstm/TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ*
element_dtype02)
'lstm/TensorArrayV2Stack/TensorListStack
lstm/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
lstm/strided_slice_3/stack
lstm/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
lstm/strided_slice_3/stack_1
lstm/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
lstm/strided_slice_3/stack_2¹
lstm/strided_slice_3StridedSlice0lstm/TensorArrayV2Stack/TensorListStack:tensor:0#lstm/strided_slice_3/stack:output:0%lstm/strided_slice_3/stack_1:output:0%lstm/strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask2
lstm/strided_slice_3
lstm/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
lstm/transpose_1/permÃ
lstm/transpose_1	Transpose0lstm/TensorArrayV2Stack/TensorListStack:tensor:0lstm/transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
lstm/transpose_1p
lstm/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm/runtime
dropout/IdentityIdentitylstm/transpose_1:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
dropout/Identity©
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes
:	È*
dtype02 
dense/Tensordot/ReadVariableOpv
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/axes}
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense/Tensordot/freew
dense/Tensordot/ShapeShapedropout/Identity:output:0*
T0*
_output_shapes
:2
dense/Tensordot/Shape
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axisï
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axisõ
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2_1x
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod|
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_1 
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod_1|
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat/axisÎ
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat¤
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stack¿
dense/Tensordot/transpose	Transposedropout/Identity:output:0dense/Tensordot/concat:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
dense/Tensordot/transpose·
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense/Tensordot/Reshape¶
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/Tensordot/MatMul|
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/Const_2
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axisÛ
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1±
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense/Tensordot
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp¨
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense/BiasAdd
dropout_1/IdentityIdentitydense/BiasAdd:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout_1/Identity®
 dense_1/Tensordot/ReadVariableOpReadVariableOp)dense_1_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02"
 dense_1/Tensordot/ReadVariableOpz
dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_1/Tensordot/axes
dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_1/Tensordot/free}
dense_1/Tensordot/ShapeShapedropout_1/Identity:output:0*
T0*
_output_shapes
:2
dense_1/Tensordot/Shape
dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_1/Tensordot/GatherV2/axisù
dense_1/Tensordot/GatherV2GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/free:output:0(dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_1/Tensordot/GatherV2
!dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_1/Tensordot/GatherV2_1/axisÿ
dense_1/Tensordot/GatherV2_1GatherV2 dense_1/Tensordot/Shape:output:0dense_1/Tensordot/axes:output:0*dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_1/Tensordot/GatherV2_1|
dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_1/Tensordot/Const 
dense_1/Tensordot/ProdProd#dense_1/Tensordot/GatherV2:output:0 dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_1/Tensordot/Prod
dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_1/Tensordot/Const_1¨
dense_1/Tensordot/Prod_1Prod%dense_1/Tensordot/GatherV2_1:output:0"dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_1/Tensordot/Prod_1
dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_1/Tensordot/concat/axisØ
dense_1/Tensordot/concatConcatV2dense_1/Tensordot/free:output:0dense_1/Tensordot/axes:output:0&dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/concat¬
dense_1/Tensordot/stackPackdense_1/Tensordot/Prod:output:0!dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/stackÆ
dense_1/Tensordot/transpose	Transposedropout_1/Identity:output:0!dense_1/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_1/Tensordot/transpose¿
dense_1/Tensordot/ReshapeReshapedense_1/Tensordot/transpose:y:0 dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_1/Tensordot/Reshape¾
dense_1/Tensordot/MatMulMatMul"dense_1/Tensordot/Reshape:output:0(dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/Tensordot/MatMul
dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense_1/Tensordot/Const_2
dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_1/Tensordot/concat_1/axiså
dense_1/Tensordot/concat_1ConcatV2#dense_1/Tensordot/GatherV2:output:0"dense_1/Tensordot/Const_2:output:0(dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_1/Tensordot/concat_1¹
dense_1/TensordotReshape"dense_1/Tensordot/MatMul:product:0#dense_1/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_1/Tensordot¤
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp°
dense_1/BiasAddBiasAdddense_1/Tensordot:output:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dense_1/BiasAdd
IdentityIdentitydense_1/BiasAdd:output:0^lstm/while*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::::2

lstm/while
lstm/while:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
Ó
while_body_12790
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
/while_lstm_cell_split_readvariableop_resource_05
1while_lstm_cell_split_1_readvariableop_resource_0-
)while_lstm_cell_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
-while_lstm_cell_split_readvariableop_resource3
/while_lstm_cell_split_1_readvariableop_resource+
'while_lstm_cell_readvariableop_resourceÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim¾
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
 *
dtype02&
$while/lstm_cell/split/ReadVariableOpï
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È:
È:
È:
È*
	num_split2
while/lstm_cell/split¿
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMulÃ
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_1Ã
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_2Ã
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_3t
while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const_1
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!while/lstm_cell/split_1/split_dim¿
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
: *
dtype02(
&while/lstm_cell/split_1/ReadVariableOpã
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:È:È:È:È*
	num_split2
while/lstm_cell/split_1´
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/BiasAddº
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/BiasAdd_1º
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/BiasAdd_2º
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/BiasAdd_3¬
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02 
while/lstm_cell/ReadVariableOp
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#while/lstm_cell/strided_slice/stack
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    È   2'
%while/lstm_cell/strided_slice/stack_1
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice/stack_2Þ
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
while/lstm_cell/strided_slice®
while/lstm_cell/MatMul_4MatMulwhile_placeholder_2&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_4¬
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/addw
while/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
while/lstm_cell/Const_2w
while/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/Const_3
while/lstm_cell/MulMulwhile/lstm_cell/add:z:0 while/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Mul£
while/lstm_cell/Add_1Addwhile/lstm_cell/Mul:z:0 while/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Add_1
'while/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'while/lstm_cell/clip_by_value/Minimum/yÙ
%while/lstm_cell/clip_by_value/MinimumMinimumwhile/lstm_cell/Add_1:z:00while/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2'
%while/lstm_cell/clip_by_value/Minimum
while/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
while/lstm_cell/clip_by_value/yÑ
while/lstm_cell/clip_by_valueMaximum)while/lstm_cell/clip_by_value/Minimum:z:0(while/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/clip_by_value°
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02"
 while/lstm_cell/ReadVariableOp_1
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    È   2'
%while/lstm_cell/strided_slice_1/stack£
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_1£
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_2ê
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_1°
while/lstm_cell/MatMul_5MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_5²
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/add_2w
while/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
while/lstm_cell/Const_4w
while/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/Const_5¥
while/lstm_cell/Mul_1Mulwhile/lstm_cell/add_2:z:0 while/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Mul_1¥
while/lstm_cell/Add_3Addwhile/lstm_cell/Mul_1:z:0 while/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Add_3
)while/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)while/lstm_cell/clip_by_value_1/Minimum/yß
'while/lstm_cell/clip_by_value_1/MinimumMinimumwhile/lstm_cell/Add_3:z:02while/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2)
'while/lstm_cell/clip_by_value_1/Minimum
!while/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!while/lstm_cell/clip_by_value_1/yÙ
while/lstm_cell/clip_by_value_1Maximum+while/lstm_cell/clip_by_value_1/Minimum:z:0*while/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2!
while/lstm_cell/clip_by_value_1¢
while/lstm_cell/mul_2Mul#while/lstm_cell/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/mul_2°
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02"
 while/lstm_cell/ReadVariableOp_2
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice_2/stack£
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2)
'while/lstm_cell/strided_slice_2/stack_1£
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_2/stack_2ê
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_2°
while/lstm_cell/MatMul_6MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_6²
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/add_4
while/lstm_cell/TanhTanhwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Tanh¥
while/lstm_cell/mul_3Mul!while/lstm_cell/clip_by_value:z:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/mul_3 
while/lstm_cell/add_5AddV2while/lstm_cell/mul_2:z:0while/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/add_5°
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02"
 while/lstm_cell/ReadVariableOp_3
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2'
%while/lstm_cell/strided_slice_3/stack£
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell/strided_slice_3/stack_1£
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_3/stack_2ê
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_3°
while/lstm_cell/MatMul_7MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_7²
while/lstm_cell/add_6AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/add_6w
while/lstm_cell/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
while/lstm_cell/Const_6w
while/lstm_cell/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/Const_7¥
while/lstm_cell/Mul_4Mulwhile/lstm_cell/add_6:z:0 while/lstm_cell/Const_6:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Mul_4¥
while/lstm_cell/Add_7Addwhile/lstm_cell/Mul_4:z:0 while/lstm_cell/Const_7:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Add_7
)while/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)while/lstm_cell/clip_by_value_2/Minimum/yß
'while/lstm_cell/clip_by_value_2/MinimumMinimumwhile/lstm_cell/Add_7:z:02while/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2)
'while/lstm_cell/clip_by_value_2/Minimum
!while/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!while/lstm_cell/clip_by_value_2/yÙ
while/lstm_cell/clip_by_value_2Maximum+while/lstm_cell/clip_by_value_2/Minimum:z:0*while/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2!
while/lstm_cell/clip_by_value_2
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Tanh_1©
while/lstm_cell/mul_5Mul#while/lstm_cell/clip_by_value_2:z:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/mul_5Ý
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3~
while/Identity_4Identitywhile/lstm_cell/mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/Identity_4~
while/Identity_5Identitywhile/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
: 
ë
c
D__inference_dropout_1_layer_call_and_return_conditional_losses_11495

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÁ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yË
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
dropout/Mul_1r
IdentityIdentitydropout/Mul_1:z:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
Ó
while_body_12521
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
/while_lstm_cell_split_readvariableop_resource_05
1while_lstm_cell_split_1_readvariableop_resource_0-
)while_lstm_cell_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
-while_lstm_cell_split_readvariableop_resource3
/while_lstm_cell_split_1_readvariableop_resource+
'while_lstm_cell_readvariableop_resourceÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim¾
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
 *
dtype02&
$while/lstm_cell/split/ReadVariableOpï
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È:
È:
È:
È*
	num_split2
while/lstm_cell/split¿
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMulÃ
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_1Ã
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_2Ã
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_3t
while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const_1
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!while/lstm_cell/split_1/split_dim¿
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
: *
dtype02(
&while/lstm_cell/split_1/ReadVariableOpã
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:È:È:È:È*
	num_split2
while/lstm_cell/split_1´
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/BiasAddº
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/BiasAdd_1º
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/BiasAdd_2º
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/BiasAdd_3¬
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02 
while/lstm_cell/ReadVariableOp
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#while/lstm_cell/strided_slice/stack
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    È   2'
%while/lstm_cell/strided_slice/stack_1
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice/stack_2Þ
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
while/lstm_cell/strided_slice®
while/lstm_cell/MatMul_4MatMulwhile_placeholder_2&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_4¬
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/addw
while/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
while/lstm_cell/Const_2w
while/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/Const_3
while/lstm_cell/MulMulwhile/lstm_cell/add:z:0 while/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Mul£
while/lstm_cell/Add_1Addwhile/lstm_cell/Mul:z:0 while/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Add_1
'while/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'while/lstm_cell/clip_by_value/Minimum/yÙ
%while/lstm_cell/clip_by_value/MinimumMinimumwhile/lstm_cell/Add_1:z:00while/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2'
%while/lstm_cell/clip_by_value/Minimum
while/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
while/lstm_cell/clip_by_value/yÑ
while/lstm_cell/clip_by_valueMaximum)while/lstm_cell/clip_by_value/Minimum:z:0(while/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/clip_by_value°
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02"
 while/lstm_cell/ReadVariableOp_1
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    È   2'
%while/lstm_cell/strided_slice_1/stack£
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_1£
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_2ê
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_1°
while/lstm_cell/MatMul_5MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_5²
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/add_2w
while/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
while/lstm_cell/Const_4w
while/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/Const_5¥
while/lstm_cell/Mul_1Mulwhile/lstm_cell/add_2:z:0 while/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Mul_1¥
while/lstm_cell/Add_3Addwhile/lstm_cell/Mul_1:z:0 while/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Add_3
)while/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)while/lstm_cell/clip_by_value_1/Minimum/yß
'while/lstm_cell/clip_by_value_1/MinimumMinimumwhile/lstm_cell/Add_3:z:02while/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2)
'while/lstm_cell/clip_by_value_1/Minimum
!while/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!while/lstm_cell/clip_by_value_1/yÙ
while/lstm_cell/clip_by_value_1Maximum+while/lstm_cell/clip_by_value_1/Minimum:z:0*while/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2!
while/lstm_cell/clip_by_value_1¢
while/lstm_cell/mul_2Mul#while/lstm_cell/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/mul_2°
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02"
 while/lstm_cell/ReadVariableOp_2
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice_2/stack£
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2)
'while/lstm_cell/strided_slice_2/stack_1£
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_2/stack_2ê
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_2°
while/lstm_cell/MatMul_6MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_6²
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/add_4
while/lstm_cell/TanhTanhwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Tanh¥
while/lstm_cell/mul_3Mul!while/lstm_cell/clip_by_value:z:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/mul_3 
while/lstm_cell/add_5AddV2while/lstm_cell/mul_2:z:0while/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/add_5°
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02"
 while/lstm_cell/ReadVariableOp_3
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2'
%while/lstm_cell/strided_slice_3/stack£
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell/strided_slice_3/stack_1£
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_3/stack_2ê
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_3°
while/lstm_cell/MatMul_7MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_7²
while/lstm_cell/add_6AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/add_6w
while/lstm_cell/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
while/lstm_cell/Const_6w
while/lstm_cell/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/Const_7¥
while/lstm_cell/Mul_4Mulwhile/lstm_cell/add_6:z:0 while/lstm_cell/Const_6:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Mul_4¥
while/lstm_cell/Add_7Addwhile/lstm_cell/Mul_4:z:0 while/lstm_cell/Const_7:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Add_7
)while/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)while/lstm_cell/clip_by_value_2/Minimum/yß
'while/lstm_cell/clip_by_value_2/MinimumMinimumwhile/lstm_cell/Add_7:z:02while/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2)
'while/lstm_cell/clip_by_value_2/Minimum
!while/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!while/lstm_cell/clip_by_value_2/yÙ
while/lstm_cell/clip_by_value_2Maximum+while/lstm_cell/clip_by_value_2/Minimum:z:0*while/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2!
while/lstm_cell/clip_by_value_2
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Tanh_1©
while/lstm_cell/mul_5Mul#while/lstm_cell/clip_by_value_2:z:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/mul_5Ý
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3~
while/Identity_4Identitywhile/lstm_cell/mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/Identity_4~
while/Identity_5Identitywhile/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
: 
ïP

D__inference_lstm_cell_layer_call_and_return_conditional_losses_10332

inputs

states
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
 *
dtype02
split/ReadVariableOp¯
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È:
È:
È:
È*
	num_split2
splite
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
MatMuli
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

MatMul_1i
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

MatMul_2i
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
: *
dtype02
split_1/ReadVariableOp£
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:È:È:È:È*
	num_split2	
split_1t
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2	
BiasAddz
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
	BiasAdd_1z
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
	BiasAdd_2z
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
	BiasAdd_3z
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
È *
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    È   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2þ
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
strided_sliceq
MatMul_4MatMulstatesstrided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

MatMul_4l
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3_
MulMuladd:z:0Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Mulc
Add_1AddMul:z:0Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value/Minimum/y
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
clip_by_value~
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
È *
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    È   2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
strided_slice_1s
MatMul_5MatMulstatesstrided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

MatMul_5r
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5e
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Mul_1e
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value_1/Minimum/y
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
clip_by_value_1g
mul_2Mulclip_by_value_1:z:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
mul_2~
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
È *
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
strided_slice_2s
MatMul_6MatMulstatesstrided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

MatMul_6r
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
add_4R
TanhTanh	add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Tanhe
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
mul_3`
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
add_5~
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
È *
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
strided_slice_3s
MatMul_7MatMulstatesstrided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

MatMul_7r
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7e
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Mul_4e
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value_2/Minimum/y
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
clip_by_value_2V
Tanh_1Tanh	add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Tanh_1i
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
mul_5^
IdentityIdentity	mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identityb

Identity_1Identity	mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity_1b

Identity_2Identity	add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*[
_input_shapesJ
H:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ::::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_namestates
þ
ä
lstm_while_body_12160&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3%
!lstm_while_lstm_strided_slice_1_0a
]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_08
4lstm_while_lstm_cell_split_readvariableop_resource_0:
6lstm_while_lstm_cell_split_1_readvariableop_resource_02
.lstm_while_lstm_cell_readvariableop_resource_0
lstm_while_identity
lstm_while_identity_1
lstm_while_identity_2
lstm_while_identity_3
lstm_while_identity_4
lstm_while_identity_5#
lstm_while_lstm_strided_slice_1_
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor6
2lstm_while_lstm_cell_split_readvariableop_resource8
4lstm_while_lstm_cell_split_1_readvariableop_resource0
,lstm_while_lstm_cell_readvariableop_resourceÍ
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2>
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeò
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype020
.lstm/while/TensorArrayV2Read/TensorListGetItemz
lstm/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/lstm_cell/Const
$lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm/while/lstm_cell/split/split_dimÍ
)lstm/while/lstm_cell/split/ReadVariableOpReadVariableOp4lstm_while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
 *
dtype02+
)lstm/while/lstm_cell/split/ReadVariableOp
lstm/while/lstm_cell/splitSplit-lstm/while/lstm_cell/split/split_dim:output:01lstm/while/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È:
È:
È:
È*
	num_split2
lstm/while/lstm_cell/splitÓ
lstm/while/lstm_cell/MatMulMatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/MatMul×
lstm/while/lstm_cell/MatMul_1MatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/MatMul_1×
lstm/while/lstm_cell/MatMul_2MatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/MatMul_2×
lstm/while/lstm_cell/MatMul_3MatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/MatMul_3~
lstm/while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/lstm_cell/Const_1
&lstm/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&lstm/while/lstm_cell/split_1/split_dimÎ
+lstm/while/lstm_cell/split_1/ReadVariableOpReadVariableOp6lstm_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
: *
dtype02-
+lstm/while/lstm_cell/split_1/ReadVariableOp÷
lstm/while/lstm_cell/split_1Split/lstm/while/lstm_cell/split_1/split_dim:output:03lstm/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:È:È:È:È*
	num_split2
lstm/while/lstm_cell/split_1È
lstm/while/lstm_cell/BiasAddBiasAdd%lstm/while/lstm_cell/MatMul:product:0%lstm/while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/BiasAddÎ
lstm/while/lstm_cell/BiasAdd_1BiasAdd'lstm/while/lstm_cell/MatMul_1:product:0%lstm/while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2 
lstm/while/lstm_cell/BiasAdd_1Î
lstm/while/lstm_cell/BiasAdd_2BiasAdd'lstm/while/lstm_cell/MatMul_2:product:0%lstm/while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2 
lstm/while/lstm_cell/BiasAdd_2Î
lstm/while/lstm_cell/BiasAdd_3BiasAdd'lstm/while/lstm_cell/MatMul_3:product:0%lstm/while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2 
lstm/while/lstm_cell/BiasAdd_3»
#lstm/while/lstm_cell/ReadVariableOpReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02%
#lstm/while/lstm_cell/ReadVariableOp¥
(lstm/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(lstm/while/lstm_cell/strided_slice/stack©
*lstm/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    È   2,
*lstm/while/lstm_cell/strided_slice/stack_1©
*lstm/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm/while/lstm_cell/strided_slice/stack_2ü
"lstm/while/lstm_cell/strided_sliceStridedSlice+lstm/while/lstm_cell/ReadVariableOp:value:01lstm/while/lstm_cell/strided_slice/stack:output:03lstm/while/lstm_cell/strided_slice/stack_1:output:03lstm/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2$
"lstm/while/lstm_cell/strided_sliceÂ
lstm/while/lstm_cell/MatMul_4MatMullstm_while_placeholder_2+lstm/while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/MatMul_4À
lstm/while/lstm_cell/addAddV2%lstm/while/lstm_cell/BiasAdd:output:0'lstm/while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/add
lstm/while/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
lstm/while/lstm_cell/Const_2
lstm/while/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm/while/lstm_cell/Const_3³
lstm/while/lstm_cell/MulMullstm/while/lstm_cell/add:z:0%lstm/while/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/Mul·
lstm/while/lstm_cell/Add_1Addlstm/while/lstm_cell/Mul:z:0%lstm/while/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/Add_1¡
,lstm/while/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,lstm/while/lstm_cell/clip_by_value/Minimum/yí
*lstm/while/lstm_cell/clip_by_value/MinimumMinimumlstm/while/lstm_cell/Add_1:z:05lstm/while/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2,
*lstm/while/lstm_cell/clip_by_value/Minimum
$lstm/while/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$lstm/while/lstm_cell/clip_by_value/yå
"lstm/while/lstm_cell/clip_by_valueMaximum.lstm/while/lstm_cell/clip_by_value/Minimum:z:0-lstm/while/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2$
"lstm/while/lstm_cell/clip_by_value¿
%lstm/while/lstm_cell/ReadVariableOp_1ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02'
%lstm/while/lstm_cell/ReadVariableOp_1©
*lstm/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    È   2,
*lstm/while/lstm_cell/strided_slice_1/stack­
,lstm/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm/while/lstm_cell/strided_slice_1/stack_1­
,lstm/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm/while/lstm_cell/strided_slice_1/stack_2
$lstm/while/lstm_cell/strided_slice_1StridedSlice-lstm/while/lstm_cell/ReadVariableOp_1:value:03lstm/while/lstm_cell/strided_slice_1/stack:output:05lstm/while/lstm_cell/strided_slice_1/stack_1:output:05lstm/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2&
$lstm/while/lstm_cell/strided_slice_1Ä
lstm/while/lstm_cell/MatMul_5MatMullstm_while_placeholder_2-lstm/while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/MatMul_5Æ
lstm/while/lstm_cell/add_2AddV2'lstm/while/lstm_cell/BiasAdd_1:output:0'lstm/while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/add_2
lstm/while/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
lstm/while/lstm_cell/Const_4
lstm/while/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm/while/lstm_cell/Const_5¹
lstm/while/lstm_cell/Mul_1Mullstm/while/lstm_cell/add_2:z:0%lstm/while/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/Mul_1¹
lstm/while/lstm_cell/Add_3Addlstm/while/lstm_cell/Mul_1:z:0%lstm/while/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/Add_3¥
.lstm/while/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?20
.lstm/while/lstm_cell/clip_by_value_1/Minimum/yó
,lstm/while/lstm_cell/clip_by_value_1/MinimumMinimumlstm/while/lstm_cell/Add_3:z:07lstm/while/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2.
,lstm/while/lstm_cell/clip_by_value_1/Minimum
&lstm/while/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&lstm/while/lstm_cell/clip_by_value_1/yí
$lstm/while/lstm_cell/clip_by_value_1Maximum0lstm/while/lstm_cell/clip_by_value_1/Minimum:z:0/lstm/while/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2&
$lstm/while/lstm_cell/clip_by_value_1¶
lstm/while/lstm_cell/mul_2Mul(lstm/while/lstm_cell/clip_by_value_1:z:0lstm_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/mul_2¿
%lstm/while/lstm_cell/ReadVariableOp_2ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02'
%lstm/while/lstm_cell/ReadVariableOp_2©
*lstm/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm/while/lstm_cell/strided_slice_2/stack­
,lstm/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2.
,lstm/while/lstm_cell/strided_slice_2/stack_1­
,lstm/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm/while/lstm_cell/strided_slice_2/stack_2
$lstm/while/lstm_cell/strided_slice_2StridedSlice-lstm/while/lstm_cell/ReadVariableOp_2:value:03lstm/while/lstm_cell/strided_slice_2/stack:output:05lstm/while/lstm_cell/strided_slice_2/stack_1:output:05lstm/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2&
$lstm/while/lstm_cell/strided_slice_2Ä
lstm/while/lstm_cell/MatMul_6MatMullstm_while_placeholder_2-lstm/while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/MatMul_6Æ
lstm/while/lstm_cell/add_4AddV2'lstm/while/lstm_cell/BiasAdd_2:output:0'lstm/while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/add_4
lstm/while/lstm_cell/TanhTanhlstm/while/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/Tanh¹
lstm/while/lstm_cell/mul_3Mul&lstm/while/lstm_cell/clip_by_value:z:0lstm/while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/mul_3´
lstm/while/lstm_cell/add_5AddV2lstm/while/lstm_cell/mul_2:z:0lstm/while/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/add_5¿
%lstm/while/lstm_cell/ReadVariableOp_3ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02'
%lstm/while/lstm_cell/ReadVariableOp_3©
*lstm/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2,
*lstm/while/lstm_cell/strided_slice_3/stack­
,lstm/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2.
,lstm/while/lstm_cell/strided_slice_3/stack_1­
,lstm/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm/while/lstm_cell/strided_slice_3/stack_2
$lstm/while/lstm_cell/strided_slice_3StridedSlice-lstm/while/lstm_cell/ReadVariableOp_3:value:03lstm/while/lstm_cell/strided_slice_3/stack:output:05lstm/while/lstm_cell/strided_slice_3/stack_1:output:05lstm/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2&
$lstm/while/lstm_cell/strided_slice_3Ä
lstm/while/lstm_cell/MatMul_7MatMullstm_while_placeholder_2-lstm/while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/MatMul_7Æ
lstm/while/lstm_cell/add_6AddV2'lstm/while/lstm_cell/BiasAdd_3:output:0'lstm/while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/add_6
lstm/while/lstm_cell/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
lstm/while/lstm_cell/Const_6
lstm/while/lstm_cell/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm/while/lstm_cell/Const_7¹
lstm/while/lstm_cell/Mul_4Mullstm/while/lstm_cell/add_6:z:0%lstm/while/lstm_cell/Const_6:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/Mul_4¹
lstm/while/lstm_cell/Add_7Addlstm/while/lstm_cell/Mul_4:z:0%lstm/while/lstm_cell/Const_7:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/Add_7¥
.lstm/while/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?20
.lstm/while/lstm_cell/clip_by_value_2/Minimum/yó
,lstm/while/lstm_cell/clip_by_value_2/MinimumMinimumlstm/while/lstm_cell/Add_7:z:07lstm/while/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2.
,lstm/while/lstm_cell/clip_by_value_2/Minimum
&lstm/while/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&lstm/while/lstm_cell/clip_by_value_2/yí
$lstm/while/lstm_cell/clip_by_value_2Maximum0lstm/while/lstm_cell/clip_by_value_2/Minimum:z:0/lstm/while/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2&
$lstm/while/lstm_cell/clip_by_value_2
lstm/while/lstm_cell/Tanh_1Tanhlstm/while/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/Tanh_1½
lstm/while/lstm_cell/mul_5Mul(lstm/while/lstm_cell/clip_by_value_2:z:0lstm/while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/mul_5ö
/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_while_placeholder_1lstm_while_placeholderlstm/while/lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype021
/lstm/while/TensorArrayV2Write/TensorListSetItemf
lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add/y}
lstm/while/addAddV2lstm_while_placeholderlstm/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm/while/addj
lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add_1/y
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm/while/add_1m
lstm/while/IdentityIdentitylstm/while/add_1:z:0*
T0*
_output_shapes
: 2
lstm/while/Identity
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations*
T0*
_output_shapes
: 2
lstm/while/Identity_1o
lstm/while/Identity_2Identitylstm/while/add:z:0*
T0*
_output_shapes
: 2
lstm/while/Identity_2
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
lstm/while/Identity_3
lstm/while/Identity_4Identitylstm/while/lstm_cell/mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/Identity_4
lstm/while/Identity_5Identitylstm/while/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/Identity_5"3
lstm_while_identitylstm/while/Identity:output:0"7
lstm_while_identity_1lstm/while/Identity_1:output:0"7
lstm_while_identity_2lstm/while/Identity_2:output:0"7
lstm_while_identity_3lstm/while/Identity_3:output:0"7
lstm_while_identity_4lstm/while/Identity_4:output:0"7
lstm_while_identity_5lstm/while/Identity_5:output:0"^
,lstm_while_lstm_cell_readvariableop_resource.lstm_while_lstm_cell_readvariableop_resource_0"n
4lstm_while_lstm_cell_split_1_readvariableop_resource6lstm_while_lstm_cell_split_1_readvariableop_resource_0"j
2lstm_while_lstm_cell_split_readvariableop_resource4lstm_while_lstm_cell_split_readvariableop_resource_0"D
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"¼
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
: 
Å
é
?__inference_lstm_layer_call_and_return_conditional_losses_11377

inputs+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identity¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimª
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
 *
dtype02 
lstm_cell/split/ReadVariableOp×
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È:
È:
È:
È*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim«
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
: *
dtype02"
 lstm_cell/split_1/ReadVariableOpË
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:È:È:È:È*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/BiasAdd¢
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/BiasAdd_1¢
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/BiasAdd_2¢
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/BiasAdd_3
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
È *
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    È   2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2º
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMulzeros:output:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/addk
lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
lstm_cell/Const_2k
lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/Const_3
lstm_cell/MulMullstm_cell/add:z:0lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Mul
lstm_cell/Add_1Addlstm_cell/Mul:z:0lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Add_1
!lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!lstm_cell/clip_by_value/Minimum/yÁ
lstm_cell/clip_by_value/MinimumMinimumlstm_cell/Add_1:z:0*lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2!
lstm_cell/clip_by_value/Minimum{
lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cell/clip_by_value/y¹
lstm_cell/clip_by_valueMaximum#lstm_cell/clip_by_value/Minimum:z:0"lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/clip_by_value
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
È *
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    È   2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Æ
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMulzeros:output:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_5
lstm_cell/add_2AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/add_2k
lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
lstm_cell/Const_4k
lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/Const_5
lstm_cell/Mul_1Mullstm_cell/add_2:z:0lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Mul_1
lstm_cell/Add_3Addlstm_cell/Mul_1:z:0lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Add_3
#lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2%
#lstm_cell/clip_by_value_1/Minimum/yÇ
!lstm_cell/clip_by_value_1/MinimumMinimumlstm_cell/Add_3:z:0,lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2#
!lstm_cell/clip_by_value_1/Minimum
lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cell/clip_by_value_1/yÁ
lstm_cell/clip_by_value_1Maximum%lstm_cell/clip_by_value_1/Minimum:z:0$lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/clip_by_value_1
lstm_cell/mul_2Mullstm_cell/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/mul_2
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
È *
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Æ
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMulzeros:output:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_6
lstm_cell/add_4AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/add_4p
lstm_cell/TanhTanhlstm_cell/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Tanh
lstm_cell/mul_3Mullstm_cell/clip_by_value:z:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/mul_3
lstm_cell/add_5AddV2lstm_cell/mul_2:z:0lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/add_5
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
È *
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Æ
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMulzeros:output:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_7
lstm_cell/add_6AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/add_6k
lstm_cell/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
lstm_cell/Const_6k
lstm_cell/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/Const_7
lstm_cell/Mul_4Mullstm_cell/add_6:z:0lstm_cell/Const_6:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Mul_4
lstm_cell/Add_7Addlstm_cell/Mul_4:z:0lstm_cell/Const_7:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Add_7
#lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2%
#lstm_cell/clip_by_value_2/Minimum/yÇ
!lstm_cell/clip_by_value_2/MinimumMinimumlstm_cell/Add_7:z:0,lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2#
!lstm_cell/clip_by_value_2/Minimum
lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cell/clip_by_value_2/yÁ
lstm_cell/clip_by_value_2Maximum%lstm_cell/clip_by_value_2/Minimum:z:0$lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/clip_by_value_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Tanh_1
lstm_cell/mul_5Mullstm_cell/clip_by_value_2:z:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/mul_5
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÝ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_11234*
condR
while_cond_11233*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimey
IdentityIdentitytranspose_1:y:0^while*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
¾
while_cond_11233
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_11233___redundant_placeholder03
/while_while_cond_11233___redundant_placeholder13
/while_while_cond_11233___redundant_placeholder23
/while_while_cond_11233___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
:
Í
ë
?__inference_lstm_layer_call_and_return_conditional_losses_12933
inputs_0+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identity¢whileF
ShapeShapeinputs_0*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimª
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
 *
dtype02 
lstm_cell/split/ReadVariableOp×
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È:
È:
È:
È*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim«
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
: *
dtype02"
 lstm_cell/split_1/ReadVariableOpË
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:È:È:È:È*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/BiasAdd¢
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/BiasAdd_1¢
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/BiasAdd_2¢
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/BiasAdd_3
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
È *
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    È   2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2º
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMulzeros:output:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/addk
lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
lstm_cell/Const_2k
lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/Const_3
lstm_cell/MulMullstm_cell/add:z:0lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Mul
lstm_cell/Add_1Addlstm_cell/Mul:z:0lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Add_1
!lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!lstm_cell/clip_by_value/Minimum/yÁ
lstm_cell/clip_by_value/MinimumMinimumlstm_cell/Add_1:z:0*lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2!
lstm_cell/clip_by_value/Minimum{
lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cell/clip_by_value/y¹
lstm_cell/clip_by_valueMaximum#lstm_cell/clip_by_value/Minimum:z:0"lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/clip_by_value
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
È *
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    È   2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Æ
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMulzeros:output:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_5
lstm_cell/add_2AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/add_2k
lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
lstm_cell/Const_4k
lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/Const_5
lstm_cell/Mul_1Mullstm_cell/add_2:z:0lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Mul_1
lstm_cell/Add_3Addlstm_cell/Mul_1:z:0lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Add_3
#lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2%
#lstm_cell/clip_by_value_1/Minimum/yÇ
!lstm_cell/clip_by_value_1/MinimumMinimumlstm_cell/Add_3:z:0,lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2#
!lstm_cell/clip_by_value_1/Minimum
lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cell/clip_by_value_1/yÁ
lstm_cell/clip_by_value_1Maximum%lstm_cell/clip_by_value_1/Minimum:z:0$lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/clip_by_value_1
lstm_cell/mul_2Mullstm_cell/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/mul_2
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
È *
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Æ
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMulzeros:output:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_6
lstm_cell/add_4AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/add_4p
lstm_cell/TanhTanhlstm_cell/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Tanh
lstm_cell/mul_3Mullstm_cell/clip_by_value:z:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/mul_3
lstm_cell/add_5AddV2lstm_cell/mul_2:z:0lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/add_5
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
È *
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Æ
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMulzeros:output:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_7
lstm_cell/add_6AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/add_6k
lstm_cell/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
lstm_cell/Const_6k
lstm_cell/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/Const_7
lstm_cell/Mul_4Mullstm_cell/add_6:z:0lstm_cell/Const_6:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Mul_4
lstm_cell/Add_7Addlstm_cell/Mul_4:z:0lstm_cell/Const_7:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Add_7
#lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2%
#lstm_cell/clip_by_value_2/Minimum/yÇ
!lstm_cell/clip_by_value_2/MinimumMinimumlstm_cell/Add_7:z:0,lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2#
!lstm_cell/clip_by_value_2/Minimum
lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cell/clip_by_value_2/yÁ
lstm_cell/clip_by_value_2Maximum%lstm_cell/clip_by_value_2/Minimum:z:0$lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/clip_by_value_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Tanh_1
lstm_cell/mul_5Mullstm_cell/clip_by_value_2:z:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/mul_5
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÝ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_12790*
condR
while_cond_12789*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimey
IdentityIdentitytranspose_1:y:0^while*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::2
whilewhile:_ [
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
û
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_11500

inputs

identity_1g
IdentityIdentityinputs*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identityv

Identity_1IdentityIdentity:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
þ
ä
lstm_while_body_11823&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3%
!lstm_while_lstm_strided_slice_1_0a
]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_08
4lstm_while_lstm_cell_split_readvariableop_resource_0:
6lstm_while_lstm_cell_split_1_readvariableop_resource_02
.lstm_while_lstm_cell_readvariableop_resource_0
lstm_while_identity
lstm_while_identity_1
lstm_while_identity_2
lstm_while_identity_3
lstm_while_identity_4
lstm_while_identity_5#
lstm_while_lstm_strided_slice_1_
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor6
2lstm_while_lstm_cell_split_readvariableop_resource8
4lstm_while_lstm_cell_split_1_readvariableop_resource0
,lstm_while_lstm_cell_readvariableop_resourceÍ
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2>
<lstm/while/TensorArrayV2Read/TensorListGetItem/element_shapeò
.lstm/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0lstm_while_placeholderElstm/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype020
.lstm/while/TensorArrayV2Read/TensorListGetItemz
lstm/while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/lstm_cell/Const
$lstm/while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2&
$lstm/while/lstm_cell/split/split_dimÍ
)lstm/while/lstm_cell/split/ReadVariableOpReadVariableOp4lstm_while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
 *
dtype02+
)lstm/while/lstm_cell/split/ReadVariableOp
lstm/while/lstm_cell/splitSplit-lstm/while/lstm_cell/split/split_dim:output:01lstm/while/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È:
È:
È:
È*
	num_split2
lstm/while/lstm_cell/splitÓ
lstm/while/lstm_cell/MatMulMatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/MatMul×
lstm/while/lstm_cell/MatMul_1MatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/MatMul_1×
lstm/while/lstm_cell/MatMul_2MatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/MatMul_2×
lstm/while/lstm_cell/MatMul_3MatMul5lstm/while/TensorArrayV2Read/TensorListGetItem:item:0#lstm/while/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/MatMul_3~
lstm/while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/lstm_cell/Const_1
&lstm/while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2(
&lstm/while/lstm_cell/split_1/split_dimÎ
+lstm/while/lstm_cell/split_1/ReadVariableOpReadVariableOp6lstm_while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
: *
dtype02-
+lstm/while/lstm_cell/split_1/ReadVariableOp÷
lstm/while/lstm_cell/split_1Split/lstm/while/lstm_cell/split_1/split_dim:output:03lstm/while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:È:È:È:È*
	num_split2
lstm/while/lstm_cell/split_1È
lstm/while/lstm_cell/BiasAddBiasAdd%lstm/while/lstm_cell/MatMul:product:0%lstm/while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/BiasAddÎ
lstm/while/lstm_cell/BiasAdd_1BiasAdd'lstm/while/lstm_cell/MatMul_1:product:0%lstm/while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2 
lstm/while/lstm_cell/BiasAdd_1Î
lstm/while/lstm_cell/BiasAdd_2BiasAdd'lstm/while/lstm_cell/MatMul_2:product:0%lstm/while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2 
lstm/while/lstm_cell/BiasAdd_2Î
lstm/while/lstm_cell/BiasAdd_3BiasAdd'lstm/while/lstm_cell/MatMul_3:product:0%lstm/while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2 
lstm/while/lstm_cell/BiasAdd_3»
#lstm/while/lstm_cell/ReadVariableOpReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02%
#lstm/while/lstm_cell/ReadVariableOp¥
(lstm/while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2*
(lstm/while/lstm_cell/strided_slice/stack©
*lstm/while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    È   2,
*lstm/while/lstm_cell/strided_slice/stack_1©
*lstm/while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm/while/lstm_cell/strided_slice/stack_2ü
"lstm/while/lstm_cell/strided_sliceStridedSlice+lstm/while/lstm_cell/ReadVariableOp:value:01lstm/while/lstm_cell/strided_slice/stack:output:03lstm/while/lstm_cell/strided_slice/stack_1:output:03lstm/while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2$
"lstm/while/lstm_cell/strided_sliceÂ
lstm/while/lstm_cell/MatMul_4MatMullstm_while_placeholder_2+lstm/while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/MatMul_4À
lstm/while/lstm_cell/addAddV2%lstm/while/lstm_cell/BiasAdd:output:0'lstm/while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/add
lstm/while/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
lstm/while/lstm_cell/Const_2
lstm/while/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm/while/lstm_cell/Const_3³
lstm/while/lstm_cell/MulMullstm/while/lstm_cell/add:z:0%lstm/while/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/Mul·
lstm/while/lstm_cell/Add_1Addlstm/while/lstm_cell/Mul:z:0%lstm/while/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/Add_1¡
,lstm/while/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2.
,lstm/while/lstm_cell/clip_by_value/Minimum/yí
*lstm/while/lstm_cell/clip_by_value/MinimumMinimumlstm/while/lstm_cell/Add_1:z:05lstm/while/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2,
*lstm/while/lstm_cell/clip_by_value/Minimum
$lstm/while/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2&
$lstm/while/lstm_cell/clip_by_value/yå
"lstm/while/lstm_cell/clip_by_valueMaximum.lstm/while/lstm_cell/clip_by_value/Minimum:z:0-lstm/while/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2$
"lstm/while/lstm_cell/clip_by_value¿
%lstm/while/lstm_cell/ReadVariableOp_1ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02'
%lstm/while/lstm_cell/ReadVariableOp_1©
*lstm/while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    È   2,
*lstm/while/lstm_cell/strided_slice_1/stack­
,lstm/while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm/while/lstm_cell/strided_slice_1/stack_1­
,lstm/while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm/while/lstm_cell/strided_slice_1/stack_2
$lstm/while/lstm_cell/strided_slice_1StridedSlice-lstm/while/lstm_cell/ReadVariableOp_1:value:03lstm/while/lstm_cell/strided_slice_1/stack:output:05lstm/while/lstm_cell/strided_slice_1/stack_1:output:05lstm/while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2&
$lstm/while/lstm_cell/strided_slice_1Ä
lstm/while/lstm_cell/MatMul_5MatMullstm_while_placeholder_2-lstm/while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/MatMul_5Æ
lstm/while/lstm_cell/add_2AddV2'lstm/while/lstm_cell/BiasAdd_1:output:0'lstm/while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/add_2
lstm/while/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
lstm/while/lstm_cell/Const_4
lstm/while/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm/while/lstm_cell/Const_5¹
lstm/while/lstm_cell/Mul_1Mullstm/while/lstm_cell/add_2:z:0%lstm/while/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/Mul_1¹
lstm/while/lstm_cell/Add_3Addlstm/while/lstm_cell/Mul_1:z:0%lstm/while/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/Add_3¥
.lstm/while/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?20
.lstm/while/lstm_cell/clip_by_value_1/Minimum/yó
,lstm/while/lstm_cell/clip_by_value_1/MinimumMinimumlstm/while/lstm_cell/Add_3:z:07lstm/while/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2.
,lstm/while/lstm_cell/clip_by_value_1/Minimum
&lstm/while/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&lstm/while/lstm_cell/clip_by_value_1/yí
$lstm/while/lstm_cell/clip_by_value_1Maximum0lstm/while/lstm_cell/clip_by_value_1/Minimum:z:0/lstm/while/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2&
$lstm/while/lstm_cell/clip_by_value_1¶
lstm/while/lstm_cell/mul_2Mul(lstm/while/lstm_cell/clip_by_value_1:z:0lstm_while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/mul_2¿
%lstm/while/lstm_cell/ReadVariableOp_2ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02'
%lstm/while/lstm_cell/ReadVariableOp_2©
*lstm/while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"      2,
*lstm/while/lstm_cell/strided_slice_2/stack­
,lstm/while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2.
,lstm/while/lstm_cell/strided_slice_2/stack_1­
,lstm/while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm/while/lstm_cell/strided_slice_2/stack_2
$lstm/while/lstm_cell/strided_slice_2StridedSlice-lstm/while/lstm_cell/ReadVariableOp_2:value:03lstm/while/lstm_cell/strided_slice_2/stack:output:05lstm/while/lstm_cell/strided_slice_2/stack_1:output:05lstm/while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2&
$lstm/while/lstm_cell/strided_slice_2Ä
lstm/while/lstm_cell/MatMul_6MatMullstm_while_placeholder_2-lstm/while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/MatMul_6Æ
lstm/while/lstm_cell/add_4AddV2'lstm/while/lstm_cell/BiasAdd_2:output:0'lstm/while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/add_4
lstm/while/lstm_cell/TanhTanhlstm/while/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/Tanh¹
lstm/while/lstm_cell/mul_3Mul&lstm/while/lstm_cell/clip_by_value:z:0lstm/while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/mul_3´
lstm/while/lstm_cell/add_5AddV2lstm/while/lstm_cell/mul_2:z:0lstm/while/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/add_5¿
%lstm/while/lstm_cell/ReadVariableOp_3ReadVariableOp.lstm_while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02'
%lstm/while/lstm_cell/ReadVariableOp_3©
*lstm/while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2,
*lstm/while/lstm_cell/strided_slice_3/stack­
,lstm/while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2.
,lstm/while/lstm_cell/strided_slice_3/stack_1­
,lstm/while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2.
,lstm/while/lstm_cell/strided_slice_3/stack_2
$lstm/while/lstm_cell/strided_slice_3StridedSlice-lstm/while/lstm_cell/ReadVariableOp_3:value:03lstm/while/lstm_cell/strided_slice_3/stack:output:05lstm/while/lstm_cell/strided_slice_3/stack_1:output:05lstm/while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2&
$lstm/while/lstm_cell/strided_slice_3Ä
lstm/while/lstm_cell/MatMul_7MatMullstm_while_placeholder_2-lstm/while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/MatMul_7Æ
lstm/while/lstm_cell/add_6AddV2'lstm/while/lstm_cell/BiasAdd_3:output:0'lstm/while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/add_6
lstm/while/lstm_cell/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
lstm/while/lstm_cell/Const_6
lstm/while/lstm_cell/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm/while/lstm_cell/Const_7¹
lstm/while/lstm_cell/Mul_4Mullstm/while/lstm_cell/add_6:z:0%lstm/while/lstm_cell/Const_6:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/Mul_4¹
lstm/while/lstm_cell/Add_7Addlstm/while/lstm_cell/Mul_4:z:0%lstm/while/lstm_cell/Const_7:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/Add_7¥
.lstm/while/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?20
.lstm/while/lstm_cell/clip_by_value_2/Minimum/yó
,lstm/while/lstm_cell/clip_by_value_2/MinimumMinimumlstm/while/lstm_cell/Add_7:z:07lstm/while/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2.
,lstm/while/lstm_cell/clip_by_value_2/Minimum
&lstm/while/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&lstm/while/lstm_cell/clip_by_value_2/yí
$lstm/while/lstm_cell/clip_by_value_2Maximum0lstm/while/lstm_cell/clip_by_value_2/Minimum:z:0/lstm/while/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2&
$lstm/while/lstm_cell/clip_by_value_2
lstm/while/lstm_cell/Tanh_1Tanhlstm/while/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/Tanh_1½
lstm/while/lstm_cell/mul_5Mul(lstm/while/lstm_cell/clip_by_value_2:z:0lstm/while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/lstm_cell/mul_5ö
/lstm/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_while_placeholder_1lstm_while_placeholderlstm/while/lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype021
/lstm/while/TensorArrayV2Write/TensorListSetItemf
lstm/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add/y}
lstm/while/addAddV2lstm_while_placeholderlstm/while/add/y:output:0*
T0*
_output_shapes
: 2
lstm/while/addj
lstm/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
lstm/while/add_1/y
lstm/while/add_1AddV2"lstm_while_lstm_while_loop_counterlstm/while/add_1/y:output:0*
T0*
_output_shapes
: 2
lstm/while/add_1m
lstm/while/IdentityIdentitylstm/while/add_1:z:0*
T0*
_output_shapes
: 2
lstm/while/Identity
lstm/while/Identity_1Identity(lstm_while_lstm_while_maximum_iterations*
T0*
_output_shapes
: 2
lstm/while/Identity_1o
lstm/while/Identity_2Identitylstm/while/add:z:0*
T0*
_output_shapes
: 2
lstm/while/Identity_2
lstm/while/Identity_3Identity?lstm/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
lstm/while/Identity_3
lstm/while/Identity_4Identitylstm/while/lstm_cell/mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/Identity_4
lstm/while/Identity_5Identitylstm/while/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm/while/Identity_5"3
lstm_while_identitylstm/while/Identity:output:0"7
lstm_while_identity_1lstm/while/Identity_1:output:0"7
lstm_while_identity_2lstm/while/Identity_2:output:0"7
lstm_while_identity_3lstm/while/Identity_3:output:0"7
lstm_while_identity_4lstm/while/Identity_4:output:0"7
lstm_while_identity_5lstm/while/Identity_5:output:0"^
,lstm_while_lstm_cell_readvariableop_resource.lstm_while_lstm_cell_readvariableop_resource_0"n
4lstm_while_lstm_cell_split_1_readvariableop_resource6lstm_while_lstm_cell_split_1_readvariableop_resource_0"j
2lstm_while_lstm_cell_split_readvariableop_resource4lstm_while_lstm_cell_split_readvariableop_resource_0"D
lstm_while_lstm_strided_slice_1!lstm_while_lstm_strided_slice_1_0"¼
[lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor]lstm_while_tensorarrayv2read_tensorlistgetitem_lstm_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
: 
Å
½
E__inference_sequential_layer_call_and_return_conditional_losses_11651

inputs

lstm_11631

lstm_11633

lstm_11635
dense_11639
dense_11641
dense_1_11645
dense_1_11647
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢lstm/StatefulPartitionedCall
lstm/StatefulPartitionedCallStatefulPartitionedCallinputs
lstm_11631
lstm_11633
lstm_11635*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_113772
lstm/StatefulPartitionedCallý
dropout/PartitionedCallPartitionedCall%lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_114242
dropout/PartitionedCall©
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_11639dense_11641*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_114672
dense/StatefulPartitionedCall
dropout_1/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_115002
dropout_1/PartitionedCallµ
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_1_11645dense_1_11647*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_115432!
dense_1/StatefulPartitionedCallê
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^lstm/StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿
É
)__inference_lstm_cell_layer_call_fn_13863

inputs
states_0
states_1
unknown
	unknown_0
	unknown_1
identity

identity_1

identity_2¢StatefulPartitionedCallÂ
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_103322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity_1

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*[
_input_shapesJ
H:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
"
_user_specified_name
states/0:RN
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
"
_user_specified_name
states/1
û
b
D__inference_dropout_1_layer_call_and_return_conditional_losses_13598

inputs

identity_1g
IdentityIdentityinputs*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identityv

Identity_1IdentityIdentity:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity_1"!

identity_1Identity_1:output:0*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò
`
'__inference_dropout_layer_call_fn_13537

inputs
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_114192
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
Ë

E__inference_sequential_layer_call_and_return_conditional_losses_11560

lstm_input

lstm_11400

lstm_11402

lstm_11404
dense_11478
dense_11480
dense_1_11554
dense_1_11556
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢!dropout_1/StatefulPartitionedCall¢lstm/StatefulPartitionedCall
lstm/StatefulPartitionedCallStatefulPartitionedCall
lstm_input
lstm_11400
lstm_11402
lstm_11404*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_111082
lstm/StatefulPartitionedCall
dropout/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_114192!
dropout/StatefulPartitionedCall±
dense/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0dense_11478dense_11480*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_114672
dense/StatefulPartitionedCall½
!dropout_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_114952#
!dropout_1/StatefulPartitionedCall½
dense_1/StatefulPartitionedCallStatefulPartitionedCall*dropout_1/StatefulPartitionedCall:output:0dense_1_11554dense_1_11556*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_115432!
dense_1/StatefulPartitionedCall°
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^dropout_1/StatefulPartitionedCall^lstm/StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!dropout_1/StatefulPartitionedCall!dropout_1/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:a ]
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
lstm_input
º$
õ
while_body_10626
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0
while_lstm_cell_10650_0
while_lstm_cell_10652_0
while_lstm_cell_10654_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor
while_lstm_cell_10650
while_lstm_cell_10652
while_lstm_cell_10654¢'while/lstm_cell/StatefulPartitionedCallÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemÎ
'while/lstm_cell/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_10650_0while_lstm_cell_10652_0while_lstm_cell_10654_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_102412)
'while/lstm_cell/StatefulPartitionedCallô
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder0while/lstm_cell/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1
while/IdentityIdentitywhile/add_1:z:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity
while/Identity_1Identitywhile_while_maximum_iterations(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_1
while/Identity_2Identitywhile/add:z:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_2·
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0(^while/lstm_cell/StatefulPartitionedCall*
T0*
_output_shapes
: 2
while/Identity_3¿
while/Identity_4Identity0while/lstm_cell/StatefulPartitionedCall:output:1(^while/lstm_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/Identity_4¿
while/Identity_5Identity0while/lstm_cell/StatefulPartitionedCall:output:2(^while/lstm_cell/StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"0
while_lstm_cell_10650while_lstm_cell_10650_0"0
while_lstm_cell_10652while_lstm_cell_10652_0"0
while_lstm_cell_10654while_lstm_cell_10654_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : :::2R
'while/lstm_cell/StatefulPartitionedCall'while/lstm_cell/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
: 
ª
¾
while_cond_12789
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_12789___redundant_placeholder03
/while_while_cond_12789___redundant_placeholder13
/while_while_cond_12789___redundant_placeholder23
/while_while_cond_12789___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
:
Å
é
?__inference_lstm_layer_call_and_return_conditional_losses_11108

inputs+
'lstm_cell_split_readvariableop_resource-
)lstm_cell_split_1_readvariableop_resource%
!lstm_cell_readvariableop_resource
identity¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2d
lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Constx
lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/split/split_dimª
lstm_cell/split/ReadVariableOpReadVariableOp'lstm_cell_split_readvariableop_resource* 
_output_shapes
:
 *
dtype02 
lstm_cell/split/ReadVariableOp×
lstm_cell/splitSplit"lstm_cell/split/split_dim:output:0&lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È:
È:
È:
È*
	num_split2
lstm_cell/split
lstm_cell/MatMulMatMulstrided_slice_2:output:0lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul
lstm_cell/MatMul_1MatMulstrided_slice_2:output:0lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_1
lstm_cell/MatMul_2MatMulstrided_slice_2:output:0lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_2
lstm_cell/MatMul_3MatMulstrided_slice_2:output:0lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_3h
lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
lstm_cell/Const_1|
lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
lstm_cell/split_1/split_dim«
 lstm_cell/split_1/ReadVariableOpReadVariableOp)lstm_cell_split_1_readvariableop_resource*
_output_shapes	
: *
dtype02"
 lstm_cell/split_1/ReadVariableOpË
lstm_cell/split_1Split$lstm_cell/split_1/split_dim:output:0(lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:È:È:È:È*
	num_split2
lstm_cell/split_1
lstm_cell/BiasAddBiasAddlstm_cell/MatMul:product:0lstm_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/BiasAdd¢
lstm_cell/BiasAdd_1BiasAddlstm_cell/MatMul_1:product:0lstm_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/BiasAdd_1¢
lstm_cell/BiasAdd_2BiasAddlstm_cell/MatMul_2:product:0lstm_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/BiasAdd_2¢
lstm_cell/BiasAdd_3BiasAddlstm_cell/MatMul_3:product:0lstm_cell/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/BiasAdd_3
lstm_cell/ReadVariableOpReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
È *
dtype02
lstm_cell/ReadVariableOp
lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
lstm_cell/strided_slice/stack
lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    È   2!
lstm_cell/strided_slice/stack_1
lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice/stack_2º
lstm_cell/strided_sliceStridedSlice lstm_cell/ReadVariableOp:value:0&lstm_cell/strided_slice/stack:output:0(lstm_cell/strided_slice/stack_1:output:0(lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
lstm_cell/strided_slice
lstm_cell/MatMul_4MatMulzeros:output:0 lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_4
lstm_cell/addAddV2lstm_cell/BiasAdd:output:0lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/addk
lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
lstm_cell/Const_2k
lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/Const_3
lstm_cell/MulMullstm_cell/add:z:0lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Mul
lstm_cell/Add_1Addlstm_cell/Mul:z:0lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Add_1
!lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2#
!lstm_cell/clip_by_value/Minimum/yÁ
lstm_cell/clip_by_value/MinimumMinimumlstm_cell/Add_1:z:0*lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2!
lstm_cell/clip_by_value/Minimum{
lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cell/clip_by_value/y¹
lstm_cell/clip_by_valueMaximum#lstm_cell/clip_by_value/Minimum:z:0"lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/clip_by_value
lstm_cell/ReadVariableOp_1ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
È *
dtype02
lstm_cell/ReadVariableOp_1
lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    È   2!
lstm_cell/strided_slice_1/stack
!lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_1
!lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_1/stack_2Æ
lstm_cell/strided_slice_1StridedSlice"lstm_cell/ReadVariableOp_1:value:0(lstm_cell/strided_slice_1/stack:output:0*lstm_cell/strided_slice_1/stack_1:output:0*lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
lstm_cell/strided_slice_1
lstm_cell/MatMul_5MatMulzeros:output:0"lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_5
lstm_cell/add_2AddV2lstm_cell/BiasAdd_1:output:0lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/add_2k
lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
lstm_cell/Const_4k
lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/Const_5
lstm_cell/Mul_1Mullstm_cell/add_2:z:0lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Mul_1
lstm_cell/Add_3Addlstm_cell/Mul_1:z:0lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Add_3
#lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2%
#lstm_cell/clip_by_value_1/Minimum/yÇ
!lstm_cell/clip_by_value_1/MinimumMinimumlstm_cell/Add_3:z:0,lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2#
!lstm_cell/clip_by_value_1/Minimum
lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cell/clip_by_value_1/yÁ
lstm_cell/clip_by_value_1Maximum%lstm_cell/clip_by_value_1/Minimum:z:0$lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/clip_by_value_1
lstm_cell/mul_2Mullstm_cell/clip_by_value_1:z:0zeros_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/mul_2
lstm_cell/ReadVariableOp_2ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
È *
dtype02
lstm_cell/ReadVariableOp_2
lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"      2!
lstm_cell/strided_slice_2/stack
!lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2#
!lstm_cell/strided_slice_2/stack_1
!lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_2/stack_2Æ
lstm_cell/strided_slice_2StridedSlice"lstm_cell/ReadVariableOp_2:value:0(lstm_cell/strided_slice_2/stack:output:0*lstm_cell/strided_slice_2/stack_1:output:0*lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
lstm_cell/strided_slice_2
lstm_cell/MatMul_6MatMulzeros:output:0"lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_6
lstm_cell/add_4AddV2lstm_cell/BiasAdd_2:output:0lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/add_4p
lstm_cell/TanhTanhlstm_cell/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Tanh
lstm_cell/mul_3Mullstm_cell/clip_by_value:z:0lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/mul_3
lstm_cell/add_5AddV2lstm_cell/mul_2:z:0lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/add_5
lstm_cell/ReadVariableOp_3ReadVariableOp!lstm_cell_readvariableop_resource* 
_output_shapes
:
È *
dtype02
lstm_cell/ReadVariableOp_3
lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2!
lstm_cell/strided_slice_3/stack
!lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2#
!lstm_cell/strided_slice_3/stack_1
!lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2#
!lstm_cell/strided_slice_3/stack_2Æ
lstm_cell/strided_slice_3StridedSlice"lstm_cell/ReadVariableOp_3:value:0(lstm_cell/strided_slice_3/stack:output:0*lstm_cell/strided_slice_3/stack_1:output:0*lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
lstm_cell/strided_slice_3
lstm_cell/MatMul_7MatMulzeros:output:0"lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/MatMul_7
lstm_cell/add_6AddV2lstm_cell/BiasAdd_3:output:0lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/add_6k
lstm_cell/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
lstm_cell/Const_6k
lstm_cell/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
lstm_cell/Const_7
lstm_cell/Mul_4Mullstm_cell/add_6:z:0lstm_cell/Const_6:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Mul_4
lstm_cell/Add_7Addlstm_cell/Mul_4:z:0lstm_cell/Const_7:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Add_7
#lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2%
#lstm_cell/clip_by_value_2/Minimum/yÇ
!lstm_cell/clip_by_value_2/MinimumMinimumlstm_cell/Add_7:z:0,lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2#
!lstm_cell/clip_by_value_2/Minimum
lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lstm_cell/clip_by_value_2/yÁ
lstm_cell/clip_by_value_2Maximum%lstm_cell/clip_by_value_2/Minimum:z:0$lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/clip_by_value_2t
lstm_cell/Tanh_1Tanhlstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/Tanh_1
lstm_cell/mul_5Mullstm_cell/clip_by_value_2:z:0lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
lstm_cell/mul_5
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counterÝ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0'lstm_cell_split_readvariableop_resource)lstm_cell_split_1_readvariableop_resource!lstm_cell_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_10965*
condR
while_cond_10964*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtimey
IdentityIdentitytranspose_1:y:0^while*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ
Á
E__inference_sequential_layer_call_and_return_conditional_losses_11583

lstm_input

lstm_11563

lstm_11565

lstm_11567
dense_11571
dense_11573
dense_1_11577
dense_1_11579
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢lstm/StatefulPartitionedCall
lstm/StatefulPartitionedCallStatefulPartitionedCall
lstm_input
lstm_11563
lstm_11565
lstm_11567*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_113772
lstm/StatefulPartitionedCallý
dropout/PartitionedCallPartitionedCall%lstm/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_114242
dropout/PartitionedCall©
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0dense_11571dense_11573*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_114672
dense/StatefulPartitionedCall
dropout_1/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dropout_1_layer_call_and_return_conditional_losses_115002
dropout_1/PartitionedCallµ
dense_1/StatefulPartitionedCallStatefulPartitionedCall"dropout_1/PartitionedCall:output:0dense_1_11577dense_1_11579*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_115432!
dense_1/StatefulPartitionedCallê
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^lstm/StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:a ]
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
lstm_input
µ
Ê
*__inference_sequential_layer_call_fn_12395

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity¢StatefulPartitionedCallÃ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_116512
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ
a
B__inference_dropout_layer_call_and_return_conditional_losses_11419

inputs
identityc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?2
dropout/Const
dropout/MulMulinputsdropout/Const:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeÂ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=2
dropout/GreaterEqual/yÌ
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
dropout/GreaterEqual
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
dropout/Cast
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
dropout/Mul_1s
IdentityIdentitydropout/Mul_1:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
´D
É
?__inference_lstm_layer_call_and_return_conditional_losses_10827

inputs
lstm_cell_10745
lstm_cell_10747
lstm_cell_10749
identity¢!lstm_cell/StatefulPartitionedCall¢whileD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice]
zeros/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
zeros/mul/yl
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: 2
	zeros/mul_
zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros/Less/yg

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: 2

zeros/Lessc
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
zeros/packed/1
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros/packed_
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros/Constv
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
zerosa
zeros_1/mul/yConst*
_output_shapes
: *
dtype0*
value
B :È2
zeros_1/mul/yr
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: 2
zeros_1/mulc
zeros_1/Less/yConst*
_output_shapes
: *
dtype0*
value
B :è2
zeros_1/Less/yo
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: 2
zeros_1/Lessg
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value
B :È2
zeros_1/packed/1
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:2
zeros_1/packedc
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
zeros_1/Const~
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2	
zeros_1u
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose/perm
	transpose	Transposeinputstranspose/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	transposeO
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:2	
Shape_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2î
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
TensorArrayV2/element_shape²
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2¿
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   27
5TensorArrayUnstack/TensorListFromTensor/element_shapeø
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02)
'TensorArrayUnstack/TensorListFromTensorx
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2ý
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask2
strided_slice_2
!lstm_cell/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_10745lstm_cell_10747lstm_cell_10749*
Tin

2*
Tout
2*
_collective_manager_ids
 *P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_lstm_cell_layer_call_and_return_conditional_losses_103322#
!lstm_cell/StatefulPartitionedCall
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   2
TensorArrayV2_1/element_shape¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
TensorArrayV2_1N
timeConst*
_output_shapes
: *
dtype0*
value	B : 2
time
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ2
while/maximum_iterationsj
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
while/loop_counter
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_10745lstm_cell_10747lstm_cell_10749*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*N
_output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *%
_read_only_resource_inputs
	
*
bodyR
while_body_10758*
condR
while_cond_10757*M
output_shapes<
:: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : : : : *
parallel_iterations 2
whileµ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÈ   22
0TensorArrayV2Stack/TensorListStack/element_shapeò
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ*
element_dtype02$
"TensorArrayV2Stack/TensorListStack
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ*
shrink_axis_mask2
strided_slice_3y
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          2
transpose_1/perm¯
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
transpose_1f
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    2	
runtime
IdentityIdentitytranspose_1:y:0"^lstm_cell/StatefulPartitionedCall^while*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::2F
!lstm_cell/StatefulPartitionedCall!lstm_cell/StatefulPartitionedCall2
whilewhile:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Á
Î
*__inference_sequential_layer_call_fn_11626

lstm_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity¢StatefulPartitionedCallÇ
StatefulPartitionedCallStatefulPartitionedCall
lstm_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_116092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
lstm_input
Á	
¢
lstm_while_cond_12159&
"lstm_while_lstm_while_loop_counter,
(lstm_while_lstm_while_maximum_iterations
lstm_while_placeholder
lstm_while_placeholder_1
lstm_while_placeholder_2
lstm_while_placeholder_3(
$lstm_while_less_lstm_strided_slice_1=
9lstm_while_lstm_while_cond_12159___redundant_placeholder0=
9lstm_while_lstm_while_cond_12159___redundant_placeholder1=
9lstm_while_lstm_while_cond_12159___redundant_placeholder2=
9lstm_while_lstm_while_cond_12159___redundant_placeholder3
lstm_while_identity

lstm/while/LessLesslstm_while_placeholder$lstm_while_less_lstm_strided_slice_1*
T0*
_output_shapes
: 2
lstm/while/Lessl
lstm/while/IdentityIdentitylstm/while/Less:z:0*
T0
*
_output_shapes
: 2
lstm/while/Identity"3
lstm_while_identitylstm/while/Identity:output:0*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
:
ïP

D__inference_lstm_cell_layer_call_and_return_conditional_losses_10241

inputs

states
states_1!
split_readvariableop_resource#
split_1_readvariableop_resource
readvariableop_resource
identity

identity_1

identity_2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Constd
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2
split/split_dim
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource* 
_output_shapes
:
 *
dtype02
split/ReadVariableOp¯
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È:
È:
È:
È*
	num_split2
splite
MatMulMatMulinputssplit:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
MatMuli
MatMul_1MatMulinputssplit:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

MatMul_1i
MatMul_2MatMulinputssplit:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

MatMul_2i
MatMul_3MatMulinputssplit:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

MatMul_3T
Const_1Const*
_output_shapes
: *
dtype0*
value	B :2	
Const_1h
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2
split_1/split_dim
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes	
: *
dtype02
split_1/ReadVariableOp£
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:È:È:È:È*
	num_split2	
split_1t
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2	
BiasAddz
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
	BiasAdd_1z
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
	BiasAdd_2z
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
	BiasAdd_3z
ReadVariableOpReadVariableOpreadvariableop_resource* 
_output_shapes
:
È *
dtype02
ReadVariableOp{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    È   2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2þ
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
strided_sliceq
MatMul_4MatMulstatesstrided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

MatMul_4l
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
addW
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
Const_2W
Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_3_
MulMuladd:z:0Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Mulc
Add_1AddMul:z:0Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Add_1w
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value/Minimum/y
clip_by_value/MinimumMinimum	Add_1:z:0 clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
clip_by_value/Minimumg
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value/y
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
clip_by_value~
ReadVariableOp_1ReadVariableOpreadvariableop_resource* 
_output_shapes
:
È *
dtype02
ReadVariableOp_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    È   2
strided_slice_1/stack
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_1
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
strided_slice_1s
MatMul_5MatMulstatesstrided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

MatMul_5r
add_2AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
add_2W
Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
Const_4W
Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_5e
Mul_1Mul	add_2:z:0Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Mul_1e
Add_3Add	Mul_1:z:0Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Add_3{
clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value_1/Minimum/y
clip_by_value_1/MinimumMinimum	Add_3:z:0"clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
clip_by_value_1/Minimumk
clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_1/y
clip_by_value_1Maximumclip_by_value_1/Minimum:z:0clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
clip_by_value_1g
mul_2Mulclip_by_value_1:z:0states_1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
mul_2~
ReadVariableOp_2ReadVariableOpreadvariableop_resource* 
_output_shapes
:
È *
dtype02
ReadVariableOp_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2
strided_slice_2/stack_1
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
strided_slice_2s
MatMul_6MatMulstatesstrided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

MatMul_6r
add_4AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
add_4R
TanhTanh	add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Tanhe
mul_3Mulclip_by_value:z:0Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
mul_3`
add_5AddV2	mul_2:z:0	mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
add_5~
ReadVariableOp_3ReadVariableOpreadvariableop_resource* 
_output_shapes
:
È *
dtype02
ReadVariableOp_3
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2
strided_slice_3/stack
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_3/stack_1
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_3/stack_2
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
strided_slice_3s
MatMul_7MatMulstatesstrided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

MatMul_7r
add_6AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
add_6W
Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2	
Const_6W
Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
Const_7e
Mul_4Mul	add_6:z:0Const_6:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Mul_4e
Add_7Add	Mul_4:z:0Const_7:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Add_7{
clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2
clip_by_value_2/Minimum/y
clip_by_value_2/MinimumMinimum	Add_7:z:0"clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
clip_by_value_2/Minimumk
clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
clip_by_value_2/y
clip_by_value_2Maximumclip_by_value_2/Minimum:z:0clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
clip_by_value_2V
Tanh_1Tanh	add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
Tanh_1i
mul_5Mulclip_by_value_2:z:0
Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
mul_5^
IdentityIdentity	mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identityb

Identity_1Identity	mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity_1b

Identity_2Identity	add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*[
_input_shapesJ
H:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ::::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_namestates:PL
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_namestates
¶
«
@__inference_dense_layer_call_and_return_conditional_losses_11467

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes
:	È*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisÑ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAddq
IdentityIdentityBiasAdd:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:::] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs

z
%__inference_dense_layer_call_fn_13581

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallý
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_114672
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
ý
`
B__inference_dropout_layer_call_and_return_conditional_losses_13532

inputs

identity_1h
IdentityIdentityinputs*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2

Identityw

Identity_1IdentityIdentity:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2

Identity_1"!

identity_1Identity_1:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs
Ð
Ó
while_body_13350
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_03
/while_lstm_cell_split_readvariableop_resource_05
1while_lstm_cell_split_1_readvariableop_resource_0-
)while_lstm_cell_readvariableop_resource_0
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor1
-while_lstm_cell_split_readvariableop_resource3
/while_lstm_cell_split_1_readvariableop_resource+
'while_lstm_cell_readvariableop_resourceÃ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   29
7while/TensorArrayV2Read/TensorListGetItem/element_shapeÔ
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype02+
)while/TensorArrayV2Read/TensorListGetItemp
while/lstm_cell/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const
while/lstm_cell/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :2!
while/lstm_cell/split/split_dim¾
$while/lstm_cell/split/ReadVariableOpReadVariableOp/while_lstm_cell_split_readvariableop_resource_0* 
_output_shapes
:
 *
dtype02&
$while/lstm_cell/split/ReadVariableOpï
while/lstm_cell/splitSplit(while/lstm_cell/split/split_dim:output:0,while/lstm_cell/split/ReadVariableOp:value:0*
T0*D
_output_shapes2
0:
È:
È:
È:
È*
	num_split2
while/lstm_cell/split¿
while/lstm_cell/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMulÃ
while/lstm_cell/MatMul_1MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_1Ã
while/lstm_cell/MatMul_2MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_2Ã
while/lstm_cell/MatMul_3MatMul0while/TensorArrayV2Read/TensorListGetItem:item:0while/lstm_cell/split:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_3t
while/lstm_cell/Const_1Const*
_output_shapes
: *
dtype0*
value	B :2
while/lstm_cell/Const_1
!while/lstm_cell/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 2#
!while/lstm_cell/split_1/split_dim¿
&while/lstm_cell/split_1/ReadVariableOpReadVariableOp1while_lstm_cell_split_1_readvariableop_resource_0*
_output_shapes	
: *
dtype02(
&while/lstm_cell/split_1/ReadVariableOpã
while/lstm_cell/split_1Split*while/lstm_cell/split_1/split_dim:output:0.while/lstm_cell/split_1/ReadVariableOp:value:0*
T0*0
_output_shapes
:È:È:È:È*
	num_split2
while/lstm_cell/split_1´
while/lstm_cell/BiasAddBiasAdd while/lstm_cell/MatMul:product:0 while/lstm_cell/split_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/BiasAddº
while/lstm_cell/BiasAdd_1BiasAdd"while/lstm_cell/MatMul_1:product:0 while/lstm_cell/split_1:output:1*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/BiasAdd_1º
while/lstm_cell/BiasAdd_2BiasAdd"while/lstm_cell/MatMul_2:product:0 while/lstm_cell/split_1:output:2*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/BiasAdd_2º
while/lstm_cell/BiasAdd_3BiasAdd"while/lstm_cell/MatMul_3:product:0 while/lstm_cell/split_1:output:3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/BiasAdd_3¬
while/lstm_cell/ReadVariableOpReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02 
while/lstm_cell/ReadVariableOp
#while/lstm_cell/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2%
#while/lstm_cell/strided_slice/stack
%while/lstm_cell/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    È   2'
%while/lstm_cell/strided_slice/stack_1
%while/lstm_cell/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice/stack_2Þ
while/lstm_cell/strided_sliceStridedSlice&while/lstm_cell/ReadVariableOp:value:0,while/lstm_cell/strided_slice/stack:output:0.while/lstm_cell/strided_slice/stack_1:output:0.while/lstm_cell/strided_slice/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2
while/lstm_cell/strided_slice®
while/lstm_cell/MatMul_4MatMulwhile_placeholder_2&while/lstm_cell/strided_slice:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_4¬
while/lstm_cell/addAddV2 while/lstm_cell/BiasAdd:output:0"while/lstm_cell/MatMul_4:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/addw
while/lstm_cell/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
while/lstm_cell/Const_2w
while/lstm_cell/Const_3Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/Const_3
while/lstm_cell/MulMulwhile/lstm_cell/add:z:0 while/lstm_cell/Const_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Mul£
while/lstm_cell/Add_1Addwhile/lstm_cell/Mul:z:0 while/lstm_cell/Const_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Add_1
'while/lstm_cell/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2)
'while/lstm_cell/clip_by_value/Minimum/yÙ
%while/lstm_cell/clip_by_value/MinimumMinimumwhile/lstm_cell/Add_1:z:00while/lstm_cell/clip_by_value/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2'
%while/lstm_cell/clip_by_value/Minimum
while/lstm_cell/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
while/lstm_cell/clip_by_value/yÑ
while/lstm_cell/clip_by_valueMaximum)while/lstm_cell/clip_by_value/Minimum:z:0(while/lstm_cell/clip_by_value/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/clip_by_value°
 while/lstm_cell/ReadVariableOp_1ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02"
 while/lstm_cell/ReadVariableOp_1
%while/lstm_cell/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    È   2'
%while/lstm_cell/strided_slice_1/stack£
'while/lstm_cell/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_1£
'while/lstm_cell/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_1/stack_2ê
while/lstm_cell/strided_slice_1StridedSlice(while/lstm_cell/ReadVariableOp_1:value:0.while/lstm_cell/strided_slice_1/stack:output:00while/lstm_cell/strided_slice_1/stack_1:output:00while/lstm_cell/strided_slice_1/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_1°
while/lstm_cell/MatMul_5MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_1:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_5²
while/lstm_cell/add_2AddV2"while/lstm_cell/BiasAdd_1:output:0"while/lstm_cell/MatMul_5:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/add_2w
while/lstm_cell/Const_4Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
while/lstm_cell/Const_4w
while/lstm_cell/Const_5Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/Const_5¥
while/lstm_cell/Mul_1Mulwhile/lstm_cell/add_2:z:0 while/lstm_cell/Const_4:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Mul_1¥
while/lstm_cell/Add_3Addwhile/lstm_cell/Mul_1:z:0 while/lstm_cell/Const_5:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Add_3
)while/lstm_cell/clip_by_value_1/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)while/lstm_cell/clip_by_value_1/Minimum/yß
'while/lstm_cell/clip_by_value_1/MinimumMinimumwhile/lstm_cell/Add_3:z:02while/lstm_cell/clip_by_value_1/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2)
'while/lstm_cell/clip_by_value_1/Minimum
!while/lstm_cell/clip_by_value_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!while/lstm_cell/clip_by_value_1/yÙ
while/lstm_cell/clip_by_value_1Maximum+while/lstm_cell/clip_by_value_1/Minimum:z:0*while/lstm_cell/clip_by_value_1/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2!
while/lstm_cell/clip_by_value_1¢
while/lstm_cell/mul_2Mul#while/lstm_cell/clip_by_value_1:z:0while_placeholder_3*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/mul_2°
 while/lstm_cell/ReadVariableOp_2ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02"
 while/lstm_cell/ReadVariableOp_2
%while/lstm_cell/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"      2'
%while/lstm_cell/strided_slice_2/stack£
'while/lstm_cell/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    X  2)
'while/lstm_cell/strided_slice_2/stack_1£
'while/lstm_cell/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_2/stack_2ê
while/lstm_cell/strided_slice_2StridedSlice(while/lstm_cell/ReadVariableOp_2:value:0.while/lstm_cell/strided_slice_2/stack:output:00while/lstm_cell/strided_slice_2/stack_1:output:00while/lstm_cell/strided_slice_2/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_2°
while/lstm_cell/MatMul_6MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_2:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_6²
while/lstm_cell/add_4AddV2"while/lstm_cell/BiasAdd_2:output:0"while/lstm_cell/MatMul_6:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/add_4
while/lstm_cell/TanhTanhwhile/lstm_cell/add_4:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Tanh¥
while/lstm_cell/mul_3Mul!while/lstm_cell/clip_by_value:z:0while/lstm_cell/Tanh:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/mul_3 
while/lstm_cell/add_5AddV2while/lstm_cell/mul_2:z:0while/lstm_cell/mul_3:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/add_5°
 while/lstm_cell/ReadVariableOp_3ReadVariableOp)while_lstm_cell_readvariableop_resource_0* 
_output_shapes
:
È *
dtype02"
 while/lstm_cell/ReadVariableOp_3
%while/lstm_cell/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"    X  2'
%while/lstm_cell/strided_slice_3/stack£
'while/lstm_cell/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2)
'while/lstm_cell/strided_slice_3/stack_1£
'while/lstm_cell/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'while/lstm_cell/strided_slice_3/stack_2ê
while/lstm_cell/strided_slice_3StridedSlice(while/lstm_cell/ReadVariableOp_3:value:0.while/lstm_cell/strided_slice_3/stack:output:00while/lstm_cell/strided_slice_3/stack_1:output:00while/lstm_cell/strided_slice_3/stack_2:output:0*
Index0*
T0* 
_output_shapes
:
ÈÈ*

begin_mask*
end_mask2!
while/lstm_cell/strided_slice_3°
while/lstm_cell/MatMul_7MatMulwhile_placeholder_2(while/lstm_cell/strided_slice_3:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/MatMul_7²
while/lstm_cell/add_6AddV2"while/lstm_cell/BiasAdd_3:output:0"while/lstm_cell/MatMul_7:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/add_6w
while/lstm_cell/Const_6Const*
_output_shapes
: *
dtype0*
valueB
 *ÍÌL>2
while/lstm_cell/Const_6w
while/lstm_cell/Const_7Const*
_output_shapes
: *
dtype0*
valueB
 *   ?2
while/lstm_cell/Const_7¥
while/lstm_cell/Mul_4Mulwhile/lstm_cell/add_6:z:0 while/lstm_cell/Const_6:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Mul_4¥
while/lstm_cell/Add_7Addwhile/lstm_cell/Mul_4:z:0 while/lstm_cell/Const_7:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Add_7
)while/lstm_cell/clip_by_value_2/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?2+
)while/lstm_cell/clip_by_value_2/Minimum/yß
'while/lstm_cell/clip_by_value_2/MinimumMinimumwhile/lstm_cell/Add_7:z:02while/lstm_cell/clip_by_value_2/Minimum/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2)
'while/lstm_cell/clip_by_value_2/Minimum
!while/lstm_cell/clip_by_value_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!while/lstm_cell/clip_by_value_2/yÙ
while/lstm_cell/clip_by_value_2Maximum+while/lstm_cell/clip_by_value_2/Minimum:z:0*while/lstm_cell/clip_by_value_2/y:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2!
while/lstm_cell/clip_by_value_2
while/lstm_cell/Tanh_1Tanhwhile/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/Tanh_1©
while/lstm_cell/mul_5Mul#while/lstm_cell/clip_by_value_2:z:0while/lstm_cell/Tanh_1:y:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/lstm_cell/mul_5Ý
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell/mul_5:z:0*
_output_shapes
: *
element_dtype02,
*while/TensorArrayV2Write/TensorListSetItem\
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add/yi
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: 2
	while/add`
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
while/add_1/yv
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: 2
while/add_1^
while/IdentityIdentitywhile/add_1:z:0*
T0*
_output_shapes
: 2
while/Identityq
while/Identity_1Identitywhile_while_maximum_iterations*
T0*
_output_shapes
: 2
while/Identity_1`
while/Identity_2Identitywhile/add:z:0*
T0*
_output_shapes
: 2
while/Identity_2
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
while/Identity_3~
while/Identity_4Identitywhile/lstm_cell/mul_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/Identity_4~
while/Identity_5Identitywhile/lstm_cell/add_5:z:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ2
while/Identity_5")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"T
'while_lstm_cell_readvariableop_resource)while_lstm_cell_readvariableop_resource_0"d
/while_lstm_cell_split_1_readvariableop_resource1while_lstm_cell_split_1_readvariableop_resource_0"`
-while_lstm_cell_split_readvariableop_resource/while_lstm_cell_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
: 

Ç
#__inference_signature_wrapper_11697

lstm_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity¢StatefulPartitionedCall¢
StatefulPartitionedCallStatefulPartitionedCall
lstm_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__wrapped_model_101102
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
lstm_input
ª
¾
while_cond_10964
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_10964___redundant_placeholder03
/while_while_cond_10964___redundant_placeholder13
/while_while_cond_10964___redundant_placeholder23
/while_while_cond_10964___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
:
ª
¾
while_cond_13080
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_13
/while_while_cond_13080___redundant_placeholder03
/while_while_cond_13080___redundant_placeholder13
/while_while_cond_13080___redundant_placeholder23
/while_while_cond_13080___redundant_placeholder3
while_identity
p

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: 2

while/Less]
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: 2
while/Identity")
while_identitywhile/Identity:output:0*U
_input_shapesD
B: : : : :ÿÿÿÿÿÿÿÿÿÈ:ÿÿÿÿÿÿÿÿÿÈ: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:.*
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ:

_output_shapes
: :

_output_shapes
:
µ
Ê
*__inference_sequential_layer_call_fn_12376

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
identity¢StatefulPartitionedCallÃ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*)
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_116092
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::::22
StatefulPartitionedCallStatefulPartitionedCall:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´
­
B__inference_dense_1_layer_call_and_return_conditional_losses_11543

inputs%
!tensordot_readvariableop_resource#
biasadd_readvariableop_resource
identity
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axisÑ
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis×
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis°
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/transpose
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
Tensordot/Reshape
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis½
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2
	Tensordot
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2	
BiasAddq
IdentityIdentityBiasAdd:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*;
_input_shapes*
(:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ
C
'__inference_dropout_layer_call_fn_13542

inputs
identityÎ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_dropout_layer_call_and_return_conditional_losses_114242
PartitionedCallz
IdentityIdentityPartitionedCall:output:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ2

Identity"
identityIdentity:output:0*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ:] Y
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ë
serving_default·
O

lstm_inputA
serving_default_lstm_input:0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿH
dense_1=
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:«æ
£.
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
#_self_saveable_object_factories
	optimizer

signatures
	trainable_variables

regularization_losses
	variables
	keras_api
p__call__
q_default_save_signature
*r&call_and_return_all_conditional_losses"¤+
_tf_keras_sequential+{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lstm_input"}}, {"class_name": "LSTM", "config": {"name": "lstm", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 128]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 1}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 12, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 8, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 128]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 128]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 128]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lstm_input"}}, {"class_name": "LSTM", "config": {"name": "lstm", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 128]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 1}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 12, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 8, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
è
cell

state_spec
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
s__call__
*t&call_and_return_all_conditional_losses"
_tf_keras_rnn_layerü
{"class_name": "LSTM", "name": "lstm", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 128]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, 128]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 200, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 1}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 128]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 128]}}

#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
u__call__
*v&call_and_return_all_conditional_losses"Ò
_tf_keras_layer¸{"class_name": "Dropout", "name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}


kernel
bias
#_self_saveable_object_factories
regularization_losses
	variables
trainable_variables
	keras_api
w__call__
*x&call_and_return_all_conditional_losses"Ñ
_tf_keras_layer·{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 12, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 200}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 200]}}

# _self_saveable_object_factories
!regularization_losses
"	variables
#trainable_variables
$	keras_api
y__call__
*z&call_and_return_all_conditional_losses"Ö
_tf_keras_layer¼{"class_name": "Dropout", "name": "dropout_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_1", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}


%kernel
&bias
#'_self_saveable_object_factories
(regularization_losses
)	variables
*trainable_variables
+	keras_api
{__call__
*|&call_and_return_all_conditional_losses"Ò
_tf_keras_layer¸{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 8, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 12}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null, 12]}}
 "
trackable_dict_wrapper
Ñ
,iter

-beta_1

.beta_2
	/decay
0learning_ratembmc%md&me1mf2mg3mhvivj%vk&vl1vm2vn3vo"
	optimizer
,
}serving_default"
signature_map
Q
10
21
32
3
4
%5
&6"
trackable_list_wrapper
 "
trackable_list_wrapper
Q
10
21
32
3
4
%5
&6"
trackable_list_wrapper
Ê
4layer_regularization_losses

5layers
	trainable_variables

regularization_losses
	variables
6non_trainable_variables
7metrics
8layer_metrics
p__call__
q_default_save_signature
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
Ï

1kernel
2recurrent_kernel
3bias
#9_self_saveable_object_factories
:regularization_losses
;	variables
<trainable_variables
=	keras_api
~__call__
*&call_and_return_all_conditional_losses"ï
_tf_keras_layerÕ{"class_name": "LSTMCell", "name": "lstm_cell", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lstm_cell", "trainable": true, "dtype": "float32", "units": 200, "activation": "tanh", "recurrent_activation": "hard_sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 1}}
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
10
21
32"
trackable_list_wrapper
5
10
21
32"
trackable_list_wrapper
¹
>layer_regularization_losses

?layers
regularization_losses
	variables
trainable_variables
@non_trainable_variables
Ametrics

Bstates
Clayer_metrics
s__call__
*t&call_and_return_all_conditional_losses
&t"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Dlayer_regularization_losses

Elayers
regularization_losses
	variables
trainable_variables
Fnon_trainable_variables
Gmetrics
Hlayer_metrics
u__call__
*v&call_and_return_all_conditional_losses
&v"call_and_return_conditional_losses"
_generic_user_object
:	È2dense/kernel
:2
dense/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
­
Ilayer_regularization_losses

Jlayers
regularization_losses
	variables
trainable_variables
Knon_trainable_variables
Lmetrics
Mlayer_metrics
w__call__
*x&call_and_return_all_conditional_losses
&x"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
Nlayer_regularization_losses

Olayers
!regularization_losses
"	variables
#trainable_variables
Pnon_trainable_variables
Qmetrics
Rlayer_metrics
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
 :2dense_1/kernel
:2dense_1/bias
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
­
Slayer_regularization_losses

Tlayers
(regularization_losses
)	variables
*trainable_variables
Unon_trainable_variables
Vmetrics
Wlayer_metrics
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
):'
 2lstm/lstm_cell/kernel
3:1
È 2lstm/lstm_cell/recurrent_kernel
":  2lstm/lstm_cell/bias
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
'
X0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
10
21
32"
trackable_list_wrapper
5
10
21
32"
trackable_list_wrapper
­
Ylayer_regularization_losses

Zlayers
:regularization_losses
;	variables
<trainable_variables
[non_trainable_variables
\metrics
]layer_metrics
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
»
	^total
	_count
`	variables
a	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
.
^0
_1"
trackable_list_wrapper
-
`	variables"
_generic_user_object
$:"	È2Adam/dense/kernel/m
:2Adam/dense/bias/m
%:#2Adam/dense_1/kernel/m
:2Adam/dense_1/bias/m
.:,
 2Adam/lstm/lstm_cell/kernel/m
8:6
È 2&Adam/lstm/lstm_cell/recurrent_kernel/m
':% 2Adam/lstm/lstm_cell/bias/m
$:"	È2Adam/dense/kernel/v
:2Adam/dense/bias/v
%:#2Adam/dense_1/kernel/v
:2Adam/dense_1/bias/v
.:,
 2Adam/lstm/lstm_cell/kernel/v
8:6
È 2&Adam/lstm/lstm_cell/recurrent_kernel/v
':% 2Adam/lstm/lstm_cell/bias/v
ö2ó
*__inference_sequential_layer_call_fn_11626
*__inference_sequential_layer_call_fn_12376
*__inference_sequential_layer_call_fn_12395
*__inference_sequential_layer_call_fn_11668À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ï2ì
 __inference__wrapped_model_10110Ç
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *7¢4
2/

lstm_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
â2ß
E__inference_sequential_layer_call_and_return_conditional_losses_12034
E__inference_sequential_layer_call_and_return_conditional_losses_12357
E__inference_sequential_layer_call_and_return_conditional_losses_11560
E__inference_sequential_layer_call_and_return_conditional_losses_11583À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ó2ð
$__inference_lstm_layer_call_fn_13515
$__inference_lstm_layer_call_fn_12955
$__inference_lstm_layer_call_fn_12944
$__inference_lstm_layer_call_fn_13504Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ß2Ü
?__inference_lstm_layer_call_and_return_conditional_losses_13224
?__inference_lstm_layer_call_and_return_conditional_losses_12664
?__inference_lstm_layer_call_and_return_conditional_losses_13493
?__inference_lstm_layer_call_and_return_conditional_losses_12933Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
'__inference_dropout_layer_call_fn_13542
'__inference_dropout_layer_call_fn_13537´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Â2¿
B__inference_dropout_layer_call_and_return_conditional_losses_13527
B__inference_dropout_layer_call_and_return_conditional_losses_13532´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ï2Ì
%__inference_dense_layer_call_fn_13581¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ê2ç
@__inference_dense_layer_call_and_return_conditional_losses_13572¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
2
)__inference_dropout_1_layer_call_fn_13603
)__inference_dropout_1_layer_call_fn_13608´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Æ2Ã
D__inference_dropout_1_layer_call_and_return_conditional_losses_13598
D__inference_dropout_1_layer_call_and_return_conditional_losses_13593´
«²§
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ñ2Î
'__inference_dense_1_layer_call_fn_13647¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ì2é
B__inference_dense_1_layer_call_and_return_conditional_losses_13638¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
5B3
#__inference_signature_wrapper_11697
lstm_input
2
)__inference_lstm_cell_layer_call_fn_13846
)__inference_lstm_cell_layer_call_fn_13863¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ð2Í
D__inference_lstm_cell_layer_call_and_return_conditional_losses_13829
D__inference_lstm_cell_layer_call_and_return_conditional_losses_13738¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 ±
 __inference__wrapped_model_10110132%&A¢>
7¢4
2/

lstm_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ">ª;
9
dense_1.+
dense_1ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¼
B__inference_dense_1_layer_call_and_return_conditional_losses_13638v%&<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
'__inference_dense_1_layer_call_fn_13647i%&<¢9
2¢/
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ»
@__inference_dense_layer_call_and_return_conditional_losses_13572w=¢:
3¢0
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
%__inference_dense_layer_call_fn_13581j=¢:
3¢0
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¾
D__inference_dropout_1_layer_call_and_return_conditional_losses_13593v@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ¾
D__inference_dropout_1_layer_call_and_return_conditional_losses_13598v@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
)__inference_dropout_1_layer_call_fn_13603i@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
)__inference_dropout_1_layer_call_fn_13608i@¢=
6¢3
-*
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¾
B__inference_dropout_layer_call_and_return_conditional_losses_13527xA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
p
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 ¾
B__inference_dropout_layer_call_and_return_conditional_losses_13532xA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
p 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 
'__inference_dropout_layer_call_fn_13537kA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
p
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
'__inference_dropout_layer_call_fn_13542kA¢>
7¢4
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
p 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈÍ
D__inference_lstm_cell_layer_call_and_return_conditional_losses_13738132¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿÈ
# 
states/1ÿÿÿÿÿÿÿÿÿÈ
p
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿÈ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿÈ
 
0/1/1ÿÿÿÿÿÿÿÿÿÈ
 Í
D__inference_lstm_cell_layer_call_and_return_conditional_losses_13829132¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿÈ
# 
states/1ÿÿÿÿÿÿÿÿÿÈ
p 
ª "v¢s
l¢i

0/0ÿÿÿÿÿÿÿÿÿÈ
GD
 
0/1/0ÿÿÿÿÿÿÿÿÿÈ
 
0/1/1ÿÿÿÿÿÿÿÿÿÈ
 ¢
)__inference_lstm_cell_layer_call_fn_13846ô132¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿÈ
# 
states/1ÿÿÿÿÿÿÿÿÿÈ
p
ª "f¢c

0ÿÿÿÿÿÿÿÿÿÈ
C@

1/0ÿÿÿÿÿÿÿÿÿÈ

1/1ÿÿÿÿÿÿÿÿÿÈ¢
)__inference_lstm_cell_layer_call_fn_13863ô132¢
y¢v
!
inputsÿÿÿÿÿÿÿÿÿ
M¢J
# 
states/0ÿÿÿÿÿÿÿÿÿÈ
# 
states/1ÿÿÿÿÿÿÿÿÿÈ
p 
ª "f¢c

0ÿÿÿÿÿÿÿÿÿÈ
C@

1/0ÿÿÿÿÿÿÿÿÿÈ

1/1ÿÿÿÿÿÿÿÿÿÈÐ
?__inference_lstm_layer_call_and_return_conditional_losses_12664132P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 Ð
?__inference_lstm_layer_call_and_return_conditional_losses_12933132P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 É
?__inference_lstm_layer_call_and_return_conditional_losses_13224132I¢F
?¢<
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 É
?__inference_lstm_layer_call_and_return_conditional_losses_13493132I¢F
?¢<
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "3¢0
)&
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
 §
$__inference_lstm_layer_call_fn_12944132P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ§
$__inference_lstm_layer_call_fn_12955132P¢M
F¢C
52
0-
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ 
$__inference_lstm_layer_call_fn_13504x132I¢F
?¢<
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ 
$__inference_lstm_layer_call_fn_13515x132I¢F
?¢<
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "&#ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈÒ
E__inference_sequential_layer_call_and_return_conditional_losses_11560132%&I¢F
?¢<
2/

lstm_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ò
E__inference_sequential_layer_call_and_return_conditional_losses_11583132%&I¢F
?¢<
2/

lstm_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Î
E__inference_sequential_layer_call_and_return_conditional_losses_12034132%&E¢B
;¢8
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Î
E__inference_sequential_layer_call_and_return_conditional_losses_12357132%&E¢B
;¢8
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "2¢/
(%
0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ©
*__inference_sequential_layer_call_fn_11626{132%&I¢F
?¢<
2/

lstm_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ©
*__inference_sequential_layer_call_fn_11668{132%&I¢F
?¢<
2/

lstm_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
*__inference_sequential_layer_call_fn_12376w132%&E¢B
;¢8
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
*__inference_sequential_layer_call_fn_12395w132%&E¢B
;¢8
.+
inputsÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 

 
ª "%"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÂ
#__inference_signature_wrapper_11697132%&O¢L
¢ 
EªB
@

lstm_input2/

lstm_inputÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ">ª;
9
dense_1.+
dense_1ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ