# PowerShell equivalent of the given Bash script

# Set script parameters
$batch_size = 64
$test_batch_size = 1000
$epochs = 14
$lr = 1.0
$gamma = 0.7
$seed = 1
$log_interval = 10
$optim_type = "Adadelta"  # Specify the optimizer

# Construct the command
$command = "python main.py --batch-size $batch_size --test-batch-size $test_batch_size --epochs $epochs --lr $lr --gamma $gamma --seed $seed --log-interval $log_interval --optim_type $optim_type"

# Execute the command
Invoke-Expression $command