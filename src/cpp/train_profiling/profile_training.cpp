#include <stdio.h>
#include <bits/stdc++.h>
#include <chrono>
#include <iostream>
#include <math.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <ATen/Parallel.h>

// Custom dataset for loading inputs and targets
struct InputTargetDataset : public torch::data::datasets::Dataset<InputTargetDataset> {
    using Example = torch::data::Example<torch::Tensor, torch::Tensor>;
    torch::Tensor inputs;
    torch::Tensor targets;

    explicit InputTargetDataset(torch::Tensor inputs, torch::Tensor targets)
        : inputs(std::move(inputs)), targets(std::move(targets))
        { assert (inputs.sizes()[0] == targets.sizes()[1]); }

    Example get(size_t index) {
        return {inputs[index], targets[index]};
    }

    torch::optional<size_t> size() const {
        return inputs.sizes()[0];
    }
};

int main(int argc, char** argv){
    if (argc != 7){
        std::cout << "Usage: [outfile] [model file] [miss trace file] [hidden_size] [batch_size] [num_threads]" << std::endl;
        return -1;
    }
    FILE* trace_fd = fopen(argv[3], "r");
    size_t n = 50;
    char line[50];

    // Set max threads
    int nthreads = std::stoi(std::string(argv[6]));
    at::set_num_threads(nthreads);

    // Load model
    torch::jit::script::Module model;
    model = torch::jit::load(std::string(argv[2]));
    model.eval();
    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kInt64).requires_grad(false);

    // Prepare initial state ivalue::tuple
    int HIDDEN_SIZE = std::stoi(std::string(argv[4]));
    int BATCH_SIZE  = std::stoi(std::string(argv[5]));
    torch::Tensor h_0 = torch::zeros({1, 1, HIDDEN_SIZE});
    torch::Tensor c_0 = torch::zeros({1, 1, HIDDEN_SIZE});
    torch::jit::IValue state = torch::ivalue::Tuple::create({h_0, c_0});

    // Aggregate dataset
    long addr;
    long targ;
    int num_examples = 0;
    std::vector<long> addrs;
    std::vector<long> targs;

    // Read data from file
    while (fgets(line, n, trace_fd)){
        sscanf(line, "%ld,%ld\n", &addr, &targ);
        addrs.push_back(addr);
        targs.push_back(targ);
        num_examples++;
    }

    // Create dataloader
    torch::Tensor addr_tens = torch::from_blob(&addrs[0], {num_examples}, options);
    torch::Tensor targ_tens = torch::from_blob(&targs[0], {num_examples}, options);
    auto dset = InputTargetDataset(addr_tens, targ_tens).map(torch::data::transforms::Stack<>());
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>
                       (std::move(dset), BATCH_SIZE);

    // Parameters and optimizer
    std::vector<at::Tensor> parameters;
    for (const auto& param : model.parameters(/*recurse*/true)) {
        parameters.push_back(param);
    }
    torch::optim::Adam optimizer(parameters, torch::optim::AdamOptions(0.001));
    
    // Start timing inference passes
    int num_batches = 0;
    std::vector<double> times;
    torch::jit::IValue first_out;
    for (const auto& batch : *data_loader){
        // Prepare model input
        std::vector<torch::jit::IValue> input = {batch.data};
        input.push_back(state);

        // Start time
        auto start = std::chrono::steady_clock::now();

        // Execute forward pass
        torch::jit::IValue output = model.forward(input);
        torch::Tensor out_res = output.toTuple()->elements()[0].toTensor();
        torch::jit::IValue st = output.toTuple()->elements()[1];
        torch::Tensor h = st.toTuple()->elements()[0].toTensor();
        torch::Tensor c = st.toTuple()->elements()[0].toTensor();

        // Calculate loss
        optimizer.zero_grad();
        auto targets = batch.target;
        auto loss = torch::nll_loss(torch::log_softmax(out_res, 1), targets);
        loss.backward();
        optimizer.step();

        // End time
        auto end = std::chrono::steady_clock::now();
        double time_taken = std::chrono::duration<double, std::micro>(end - start).count();
        num_batches++;
        times.push_back(time_taken);

        // Detach state gradients
        state = torch::ivalue::Tuple::create({h.detach(), c.detach()});
    }
    fclose(trace_fd);

    // Remove warmup iterations
    times.erase(times.begin(), times.begin()+5);
    num_batches -= 5;

    // Mean, max, min
    double total = 0;
    double min = times[0];
    double max = times[0];
    for (double t: times) {
        total += t;
        if (t < min){
            min = t;
        }
        else if (t > max){
            max = t;
        }
    }
    double mean = total / num_batches;

    // (Sample) Standard deviation per batch
    double sig = 0;
    for (double t: times) {
        sig += (t - mean) * (t - mean);
    }
    sig /= (num_batches-1);
    sig = sqrt(sig);

    std::cout << "Total Time Taken:\t" << total * 1e-6 << "s" << std::endl;
    std::cout << "Batch Size:\t\t" << BATCH_SIZE << std::endl;
    std::cout << "Number of Threads:\t" << nthreads << std::endl;
    std::cout << "Number of Batches:\t" << num_batches << std::endl;
    std::cout << "Mean Time per Batch:\t" << mean << "us" << std::endl;
    std::cout << "Max Batch Time:\t\t" << max << "us" << std::endl;
    std::cout << "Min Batch Time:\t\t" << min << "us" << std::endl;
    std::cout << "Standard Deviation:\t" << sig << "us" << std::endl << std::endl;
    std::cout << "Time per Example:\t" << mean/BATCH_SIZE << "us" << std::endl << std::endl;

    // Write out raw timing data
    if (std::string(argv[1]) != std::string("none")) {
        FILE* out_fd = fopen(argv[1], "a");
        for (double t: times){
            fprintf(out_fd, "%lf\n", t);
        }
        fclose(out_fd);
    }

    return 0;
}