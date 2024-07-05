#######################################################################
#                              Settings                               #
#######################################################################
RESUME_STEP = -1
MAX_PROMPT_LEN = 1024
MAX_ANSWER_LEN = 10
MAX_PRETRAIN_LEN = 8192

PROMPT_BATCH_SIZE = 256
PRETRAIN_BATCH_SIZE = 32  # 0

GENERATE_MICRO_BATCH_SIZE = 16
INFER_MICRO_BATCH_SIZE = 8
TRAIN_MICRO_BATCH_SIZE = 1

ZERO_STAGE = 3
POLICY_DP_SIZE = 2
CRITIC_DP_SIZE = 2
POLICY_GRADIENT_ACC_STEP = (PROMPT_BATCH_SIZE + PRETRAIN_BATCH_SIZE
                            ) // POLICY_DP_SIZE // TRAIN_MICRO_BATCH_SIZE
CRITIC_GRADIENT_ACC_STEP = PROMPT_BATCH_SIZE // CRITIC_DP_SIZE // TRAIN_MICRO_BATCH_SIZE  # noqa: E501

# checkout generate config
assert PROMPT_BATCH_SIZE % GENERATE_MICRO_BATCH_SIZE == 0
assert PROMPT_BATCH_SIZE % POLICY_DP_SIZE == 0
# checkout infer config
assert PROMPT_BATCH_SIZE % (INFER_MICRO_BATCH_SIZE * POLICY_DP_SIZE) == 0
assert PROMPT_BATCH_SIZE % (INFER_MICRO_BATCH_SIZE * CRITIC_DP_SIZE) == 0
# checkout learn config
assert (PROMPT_BATCH_SIZE + PRETRAIN_BATCH_SIZE) % (TRAIN_MICRO_BATCH_SIZE *
                                                    POLICY_DP_SIZE) == 0
assert (PROMPT_BATCH_SIZE) % (TRAIN_MICRO_BATCH_SIZE * CRITIC_DP_SIZE) == 0

MODEL_DTYPE = 'auto'

tokenizer_config = dict(
    pad_token_id=0,
    eos_token_id=92542,
    padding_side='left',
)

rollout_config = dict(
    policy_micro_bs=GENERATE_MICRO_BATCH_SIZE,
    reward_micro_bs=GENERATE_MICRO_BATCH_SIZE,
    max_new_tokens=MAX_ANSWER_LEN,
    write_to_file=True,
    resume_step=RESUME_STEP,
    generate_kwargs={
        'do_sample': True,
        'temperature': 1.0,
        'top_k': 0,
        'top_p': 0.9,
        'min_new_tokens': 1,
        'num_beams': 1,
        'early_stopping': True,
        'eos_token_id': 92542,
        'pad_token_id': 0,
    },
)

repeater_config = dict(
    policy_micro_bs=INFER_MICRO_BATCH_SIZE,
    critic_micro_bs=INFER_MICRO_BATCH_SIZE,
    ref_micro_bs=INFER_MICRO_BATCH_SIZE,
    kl_coeff=0.01,
    gamma=1.0,
    gae_lambda=0.99,
    clip_reward_min=-5,
    clip_reward_max=5,
    norm_rewards=True,
)

train_config = dict(
    policy_micro_bs=TRAIN_MICRO_BATCH_SIZE,
    critic_micro_bs=TRAIN_MICRO_BATCH_SIZE,
    ppo_loss_weight=1.0,
    pretrain_loss_weight=0.5,
    critic_warmup_step=0,
    save_interval=40,
    max_train_step=400,
    resume_step=RESUME_STEP,
)

model_configs = dict(
    policy=dict(
        model_path='/fs-computility/llm/shared/models/internlm2-chat-1_8b-sft/Shanghai_AI_Laboratory_20240701/internlm2-chat-1_8b-sft/',
        model_type='policy',
        trainer_config=dict(
            torch_dtype=MODEL_DTYPE,
            trainer_type='huggingface',
            use_flash_attn=True,
            gradient_checkpointing=True,
            train_kwargs=dict(
                micro_bsz=1,
                lr=1e-6,
                total_steps=1e9,
                lr_decay_rate=1,
            ),
            parallel=dict(
                data=dict(size=POLICY_DP_SIZE, mode='deepspeed'),
                tensor=dict(size=1, mode='1d'),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=dict(size=1),
            ),
            # parallel=dict(
            #     data=dict(size=1, mode='deepspeed'),
            #     tensor=dict(size=1, mode='1d'),
            #     pipeline=dict(size=1, interleaved_overlap=False),
            #     sequence=dict(size=2),
            # ),
            envs=dict(
                ACCELERATE_USE_DEEPSPEED="true",
            ),
            deepspeed_config={
                'zero_optimization': {
                    'stage': ZERO_STAGE,
                    'offload_param': {
                        'device': 'none'
                    },
                    'reduce_bucket_size': 'auto',
                    'zero_hpz_partition_size': 1,
                    'zero_quantized_weights': False,
                    'zero_quantized_gradients': False,
                    'stage3_gather_16bit_weights_on_model_save': True,
                },
                'bf16': {
                    'enabled': True
                },
                'gradient_clipping': 1.0,
                'prescale_gradients': False,
                'wall_clock_breakdown': False,
                'data_types': {
                    'grad_accum_dtype': 'fp32'
                },
                'train_micro_batch_size_per_gpu': TRAIN_MICRO_BATCH_SIZE,
                'gradient_accumulation_steps': POLICY_GRADIENT_ACC_STEP,
                'train_batch_size': PROMPT_BATCH_SIZE + PRETRAIN_BATCH_SIZE,
            },
        ),
        generator_config=dict(
            shared_with_trainer=False,
            generator_type='vllm',
            parallel=dict(
                data=dict(size=1, mode='ddp'),
                tensor=dict(size=2, mode='1d'),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=dict(size=1),
            ),
        ),
    ),
    critic=dict(
        model_path='/fs-computility/llm/shared/marl/models/internlm2/1.8B/hf/R-Luyou-1B-8k-D20240130-v1/819_hf_bf16/',
        model_type='critic',
        trainer_config=dict(
            torch_dtype=MODEL_DTYPE,
            trainer_type='huggingface',
            use_flash_attn=True,
            gradient_checkpointing=False,
            train_kwargs=dict(
                micro_bsz=1,
                lr=5e-6,
                total_steps=1e9,
                lr_decay_rate=1,
            ),
            parallel=dict(
                data=dict(size=CRITIC_DP_SIZE, mode='deepspeed'),
                tensor=dict(size=1, mode='1d'),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=dict(size=1),
            ),
            deepspeed_config={
                'zero_optimization': {
                    'stage': ZERO_STAGE,
                    'offload_param': {
                        'device': 'none'
                    },
                    'reduce_bucket_size': 'auto',
                    'zero_hpz_partition_size': 1,
                    'zero_quantized_weights': False,
                    'zero_quantized_gradients': False
                },
                'bf16': {
                    'enabled': True
                },
                'gradient_clipping': 1.0,
                'prescale_gradients': False,
                'wall_clock_breakdown': False,
                'data_types': {
                    'grad_accum_dtype': 'fp32'
                },
                'train_micro_batch_size_per_gpu': TRAIN_MICRO_BATCH_SIZE,
                'gradient_accumulation_steps': CRITIC_GRADIENT_ACC_STEP,
                'train_batch_size': PROMPT_BATCH_SIZE,
            },
        ),
    ),
    reference=dict(
        model_path='/fs-computility/llm/shared/models/internlm2-chat-1_8b-sft/Shanghai_AI_Laboratory_20240701/internlm2-chat-1_8b-sft/',
        model_type='reference',
        trainer_config=dict(
            torch_dtype=MODEL_DTYPE,
            trainer_type='huggingface',
            use_flash_attn=True,
            parallel=dict(
                data=dict(size=1, mode='ddp'),
                tensor=dict(size=1, mode='1d'),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=dict(size=1),
            ),
        ),
    ),
    reward=dict(
        model_path='/fs-computility/llm/shared/marl/models/internlm2/1.8B/hf/R-Luyou-1B-8k-D20240130-v1/819_hf_bf16/',
        model_type='reward',
        trainer_config=dict(
            torch_dtype=MODEL_DTYPE,
            trainer_type='huggingface',
            use_flash_attn=True,
            parallel=dict(
                data=dict(size=1, mode='ddp'),
                tensor=dict(size=1, mode='1d'),
                pipeline=dict(size=1, interleaved_overlap=False),
                sequence=dict(size=1),
            ),
        ),
    ),
)

prompt_dataset_config = dict(
    samples_each_epoch=PROMPT_BATCH_SIZE,
    max_len=MAX_PROMPT_LEN,
    message_type='prompt',
    random_seed=1024,
    sample_strategy='in_batch',  # 'in_data'
    message_datasets=[
        "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/Anthropic_hh-rlhf_harmless-base-train.json::0.39471242181845173",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/Anthropic_hh-rlhf_helpful-base-train.json::1.0169169094857184",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/non_toxic_single_turn_tie_both_good-train.json::0.11121731222508861",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/non_toxic_single_turn_tie_both_bad-train.json::0.20090569959726062",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/split_0_2_prompt.json::0.26883316939180785",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/toxic_single_turn-train.json::0.025890388077429893",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/zephyr-ultrachat-200k_ppo_train_1t.json::0.9503349974944786",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/lmsys-chat-english-chat-format-100char-1t.json::0.8760045284979863",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/airoboros_reward.json::0.4921029676509345",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/out_instinwild_cn_origin_18_26-rd.json::0.7018707893320465",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/split_0_prompt-refined.json::0.07822794677158923",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/indomain_writing_2k.json::0.09776173419201574",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/zhihu_177k_outline_to_artical-with-sys.json::0.7523988047734822",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/subeval_writing_prompt_only_v2.json::0.023941649189881405",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/gaokao_essay_prompt.json::0.046027356582097584",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/wangrui6_Zhihu-KOL-train.json::0.46686215920268737",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/0801-train.json::0.07655759915369054",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/shezheng_52230_20230905_prompts-train-rd.json::0.43619272099626955[REWARD_META]:cn-safety",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/yuqing_5817_20230831_prompts-train-rd.json::0.048579276553887274[REWARD_META]:cn-safety",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/shezheng_adv_7549_20230913-train-rd.json::0.06300922403073439",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/yuqing_adv_5817_20230913-train-rd.json::0.04760490711011303",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/lingdaoren_adv_4963_20230913-train-rd.json::0.04138750208793452",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/openai_summarize_from_feedback-train.json::0.17232419591321615[META]:summarization",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/puyu_chat_format_v2-train.json::0.16703476178987028[REWARD_META]:puyu",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/identity_200_sft_for_ppo_prompt_only.json::0.009233310443384496[REWARD_META]:puyu",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/COIG-0906-train.json::0.041712291902525934",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/ANLI-0904-train.json::0.041712291902525934",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/SFT6W-prompts-train.json::0.0834709823499935",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/if_cn_en_gpt4vs1and7b.json::0.46194391343887453",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/prm800k_ppo_prompt_1212.json::0.3140253521649561[META]:latex",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/maxmin_sample200_prompt_only.json::0.009279708988326125",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/gsm8k_sample200_prompt_only.json::0.009279708988326125",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/reward_patch_20240103_prompt_only.json::0.004732651584046324",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/haochen_data.json::0.10165921196711272",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/data_reflow_2w.json::0.5321449119355617",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/retrieval_refined_bench_no_alpaca.json::0.24414914348286038",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/gsm8k_ci.json::0.12318813682002933",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/math_ci.json::0.14402108349882148",
        # "/fs-computility/llm/shared/lishuaibin/datasets/prompt_datas/airoboros_reward_ocra_math.json::0.32274827861398264",
    ])

pretrain_dataset_config = dict(
    samples_each_epoch=PRETRAIN_BATCH_SIZE,
    max_len=MAX_PRETRAIN_LEN,
    message_type='pretrain',
    random_seed=1024,
    sample_strategy='in_batch',  # 'in_data'
    message_datasets=[
        './examples/rlhf/demo_datas/pretrain_data.json::0.01',
        '[HF]Anthropic/hh-rlhf/helpful-base::0.5',
        '[HF]HuggingFaceH4/summarize_from_feedback::0.5',
    ],
)
