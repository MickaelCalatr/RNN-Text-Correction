import random
import re
import numpy as np

# ## Clean all the line of the dataset
def clean_text(text):
    '''Remove unwanted characters and extra spaces from the text'''
    text = re.sub(r'\n', ' ', text)
    return text

def noise_maker(sentence, threshold):
    '''Relocate, remove, or add characters to create spelling mistakes'''

    letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m',
               'n','o','p','q','r','s','t','u','v','w','x','y','z',]
    noisy_sentence = []
    i = 0
    while i < len(sentence):
        random = np.random.uniform(0,1,1)
        # Most characters will be correct since the threshold value is high
        if random < threshold:
            noisy_sentence.append(sentence[i])
        else:
            new_random = np.random.uniform(0,1,1)
            # ~33% chance characters will swap locations
            if new_random > 0.67:
                if i == (len(sentence) - 1) or (sentence[i + 1].isdigit() and sentence[i].isdigit() and sentence[i].islower() and sentence[i + 1].islower()):
                    # If last character in sentence, it will not be typed
                    continue
                else:
                    # if any other character, swap order with following character
                    noisy_sentence.append(sentence[i+1])
                    noisy_sentence.append(sentence[i])
                    i += 1
            # ~33% chance an extra lower case letter will be added to the sentence
            elif new_random < 0.33:
                random_letter = np.random.choice(letters, 1)[0]
                noisy_sentence.append(random_letter)
                noisy_sentence.append(sentence[i])
            # ~33% chance a character will not be typed
            else:
                pass
        i += 1
    return noisy_sentence


# ## Shuffle a line to mix the characteristics
def shuffle_line(line):
    data = line.split(" ")
    random.shuffle(data)
    tmp_line = ' '.join(data)
    result = []
    for c in tmp_line:
        result.append(c)
    return result

def normal(line):
    result = []
    for c in line:
        result.append(c)
    return result

# ## Dataset augmentation and cleaning
def dataset_augmentation(raw_dataset):
    data = []
    labels = []
    i = 0
    for elements in raw_dataset:
        # line = elements['line']
        # label = elements['label']
        label = elements.split(';;')[0]
        line = elements.split(';;')[1]
        if len(label) > 5:
            cleaned_line = clean_text(line)
            cleaned_label = clean_text(label)
            data.append(cleaned_line)
            labels.append(cleaned_label)

            if i % 2:
                data.append(shuffle_line(cleaned_line))
                labels.append(cleaned_label)

            if i % 5:
                data.append(noise_maker(cleaned_line, 0.85))
                labels.append(cleaned_label)
            i += 1
            # data.append(noise_maker(cleaned_line, 0.50))
            # labels.append(normal(cleaned_label))
            #
            # data.append(noise_maker(cleaned_line, 0.95))
            # labels.append(normal(cleaned_label))
    return len(labels), data, labels
