from Sources.Configuration import config

def train(model, epochs, log_string):
    '''Train the RNN'''

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Used to determine when to stop the training early
        testing_loss_summary = []

        # Keep track of which batch iteration is being trained
        iteration = 0
        stop_early = 0

        display_step = config.display_step
        stop = 3 # If the batch_loss_testing does not decrease in 3 consecutive checks, stop training
        #per_epoch = config.test_step # Test the model 3 times per epoch
        testing_check = 10#(len(training_sorted)//batch_size//per_epoch)-1

        print()
        print("Training Model: {}".format(log_string))

        train_writer = tf.summary.FileWriter('./logs/1/train/{}'.format(log_string), sess.graph)
        test_writer = tf.summary.FileWriter('./logs/1/test/{}'.format(log_string))
        for epoch_i in range(1, epochs+1):
            batch_loss = 0
            batch_time = 0
            i = 0
            for batch_i, (input_batch, target_batch, input_length, target_length) in enumerate(
                    get_batches(training_sorted, batch_size, threshold)):
                start_time = time.time()

                summary, loss, _ = sess.run([model.merged, model.cost,model.train_op], {model.inputs: input_batch, model.targets: target_batch, model.inputs_length: input_length, model.targets_length: target_length, model.keep_prob: keep_probability})


                batch_loss += loss
                end_time = time.time()
                batch_time += end_time - start_time

                # Record the progress of training
                train_writer.add_summary(summary, iteration)

                iteration += 1

                if batch_i % display_step == 0 and batch_i > 0:
                    print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                          .format(epoch_i,
                                  epochs,
                                  batch_i,
                                  len(training_sorted) // batch_size,
                                  batch_loss / display_step,
                                  batch_time))
                    batch_loss = 0
                    batch_time = 0

                #### Testing ####
                if batch_i % testing_check == 0 and batch_i > 0:
                    batch_loss_testing = 0
                    batch_time_testing = 0
                    for batch_i, (input_batch, target_batch, input_length, target_length) in enumerate(
                            get_batches(testing_sorted, batch_size, threshold)):
                        start_time_testing = time.time()
                        summary, loss = sess.run([model.merged,
                                                  model.cost],
                                                     {model.inputs: input_batch,
                                                      model.targets: target_batch,
                                                      model.inputs_length: input_length,
                                                      model.targets_length: target_length,
                                                      model.keep_prob: 1})

                        batch_loss_testing += loss
                        end_time_testing = time.time()
                        batch_time_testing += end_time_testing - start_time_testing

                        # Record the progress of testing
                        test_writer.add_summary(summary, iteration)

                    n_batches_testing = batch_i + 1
                    print('Testing Loss: {:>6.3f}, Seconds: {:>4.2f}'
                          .format(batch_loss_testing / n_batches_testing,
                                  batch_time_testing))

                    batch_time_testing = 0

                    # If the batch_loss_testing is at a new minimum, save the model
                    testing_loss_summary.append(batch_loss_testing)
                    if batch_loss_testing <= min(testing_loss_summary):
                        print('New Record!')
                        stop_early = 0
                        checkpoint = "./{}.ckpt".format(log_string)
                        saver = tf.train.Saver()
                        saver.save(sess, checkpoint)
                    else:
                        print("No Improvement.")
                        stop_early += 1
                        if stop_early == stop:
                            break

                if stop_early == stop:
                    print("Stopping Training.")
                    break
