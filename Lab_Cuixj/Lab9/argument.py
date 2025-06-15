def dqn_arguments(parser):
    """
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    """
    parser.add_argument('--env_name', default="CartPole-v1", help='environment name')

    parser.add_argument("--seed", default=11037, type=int)
    parser.add_argument("--hidden_size", default=32, type=int)
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    # parser.add_argument("--grad_norm_clip", default=10, type=float)
    parser.add_argument("--epsilon", default=0.2, type=float)
    parser.add_argument("--tau", default=1., type=float)

    # parser.add_argument("--test", default=False, type=bool)
    parser.add_argument("--use_cuda", default=True, type=bool)
    parser.add_argument("--n_iter", default=5000, type=int)
    parser.add_argument("--max_episode_len", default=int(500), type=int)
    parser.add_argument("--update_step", default=int(4), type=int)

    parser.add_argument("--buffer_size", default=int(500), type=int)
    parser.add_argument("--batch_size", default=int(32), type=int)
    parser.add_argument("--model_path", default="./model/best.pt", type=str)
    parser.add_argument("--log_path", default="./log/log.txt", type=str)
    parser.add_argument("--activation", default="relu", type=str)
    parser.add_argument("--dropout", default="0.0", type=float)
    parser.add_argument("--num_layers", default="2", type=str)
    parser.add_argument("--learning_rate", default="0.01", type=float)
    parser.add_argument("--epsilon_start", default="0.2", type=float)
    parser.add_argument("--save_dir", default="model.pt", type=str)
    parser.add_argument("--total_timesteps", default="50000 ", type=int)
    parser.add_argument("--target_update", default="500 ", type=int)
    parser.add_argument("--epsilon_min", default="0.01", type=float)
    parser.add_argument("--epsilon_decay", default="0.95", type=float)
    parser.add_argument("--save_interval", default="50", type=int)
    parser.add_argument("--grad_clip", default="10.0", type=float)

    return parser


