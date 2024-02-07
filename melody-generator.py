

import tensorflow as tf
import numpy as np


class MelodyGenerator:
    """
    Class to generate melodies using a trained Transformer model.

    This class encapsulates the inference logic for generating melodies
    based on a starting sequence.
    """

    def __init__(self, transformer, max_length=500):
        """
        Initializes the MelodyGenerator.

        Parameters:
            transformer (Transformer): The trained Transformer model.
            tokenizer (Tokenizer): Tokenizer used for encoding melodies.
            max_length (int): Maximum length of the generated melodies.
        """
        self.transformer = transformer
        self.max_length = max_length

    def generate(self, start_sequence):
        """
        Generates a melody based on a starting sequence.

        Parameters:
            start_sequence (list of str): The starting sequence of the melody.

        Returns:
            str: The generated melody.
        """
        input_tensor = self._get_input_tensor(start_sequence)

        num_notes_to_generate = self.max_length - len(input_tensor)

        recent_token_sequence = get_recent_token_sequence(input_tensor)
        print("recent_token_sequence")
        print(recent_token_sequence)

        

        for _ in range(num_notes_to_generate):

            prediction = 0

            # Create masks
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(input_tensor, input_tensor)
            predictions = self.transformer(
                input_tensor, input_tensor, False, None, None, None
            )
            prediction, predicted_notes = self._get_note_with_highest_score(prediction, predictions)
            input_tensor = self._append_predicted_note(input_tensor, predicted_notes)
            
            recent_token_sequence = get_recent_token_sequence(input_tensor)
            print("recent_token_sequence")
            print(recent_token_sequence)
            
            repeted_paterns = self.calculate_repetition_penalty(recent_token_sequence)
            max_index = tf.argmax(predictions, axis=-1)
            print("max_index")
            print(max_index)

            while repeted_paterns > 0:
                # Remove the last item in each row
                input_tensor = tf.slice(input_tensor, [0, 0, 0], [-1, -1, input_tensor.shape[-1]-1])
                

                prediction, predicted_notes = self._get_note_with_highest_score(prediction, predictions)
                
                input_tensor = self._append_predicted_note(input_tensor, predicted_notes)

                recent_token_sequence = get_recent_token_sequence(input_tensor)
                print("recent_token_sequence")
                print(recent_token_sequence)
                
                repeted_paterns = self.calculate_repetition_penalty(recent_token_sequence)
                max_index = tf.argmax(predictions, axis=-1)
                print("max_index")
                print(max_index)
                
                
            
            print("input_tensor")
            print(input_tensor)
            
            
            
        generated_melody = self._decode_generated_sequence(input_tensor)
        print(f"Generated melody: \n{generated_melody}")
        generated_melody = generated_melody.tolist()

        return generated_melody
    

    def _get_input_tensor(self, start_sequence):
        """
        Gets the input tensor for the Transformer model.

        Parameters:
            start_sequence (list of str): The starting sequence of the melody.

        Returns:
            input_tensor (tf.Tensor): The input tensor for the model.
        """
        input_sequence = start_sequence
        input_tensor = tf.convert_to_tensor(input_sequence, dtype=tf.int64)
        return input_tensor

    def _get_note_with_highest_score(self, prediction, predictions):
        """
        Gets the note with the highest score from the predictions.

        Parameters:
            predictions (tf.Tensor): The predictions from the model.

        Returns:
            predicted_note (int): The index of the predicted note.
        """
        prediction += 1
        print("prediction number:")
        print(prediction)
        latest_predictions = predictions[:, :, -prediction, :]
        print("latest_predictions")
        print(latest_predictions)


        temperature = 5

        # Apply temperature to the logits
        probabilities = self.apply_temperature(latest_predictions, temperature)

        predicted_note_indeces = tf.argmax(probabilities[[0]], axis=1)

        predicted_note1 = predicted_note_indeces.numpy()[0]

        predicted_note2 = predicted_note_indeces.numpy()[1]

        predicted_notes = [[predicted_note1], [predicted_note2]]

        return prediction, predicted_notes
    
    def apply_temperature(self, logits, temperature):
        # Ensure that temperature is greater than 0
        temperature = max(1e-8, temperature)

        # Apply temperature scaling to the logits
        logits /= temperature

        # Softmax to get probabilities
        probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)

        return tf.convert_to_tensor(probabilities)

    def _append_predicted_note(self, input_tensor, predicted_notes):
        """
        Appends the predicted note to the input tensor.

        Parameters:
            input_tensor (tf.Tensor): The input tensor for the model.

        Returns:
            (tf.Tensor): The input tensor with the predicted note
        """
        return tf.concat([input_tensor, [predicted_notes]], axis=-1)
    

    def _decode_generated_sequence(self, generated_sequence):
        """
        Decodes the generated sequence of notes.

        Parameters:
            generated_sequence (tf.Tensor): Tensor with note indexes generated.

        Returns:
            generated_melody (str): The decoded sequence of notes.
        """
        generated_melody = generated_sequence.numpy()

        return generated_melody
    
    def calculate_repetition_penalty(self, recent_token_sequence):
        
        # Check for repeated patterns along the last axis (axis=-1)
        repeated_patterns = tf.reduce_sum(
            tf.cast(
                tf.math.reduce_all(
                    tf.equal(
                        recent_token_sequence[:, :, 2:],
                        recent_token_sequence[:, :, :-2],
                    ),
                    axis=-1,
                ),
                tf.float32,
            )
        )

        print("repeated_patterns")
        print(np.array(repeated_patterns))

        return repeated_patterns

def create_masks(input, target):
    enc_padding_mask = create_padding_mask(input)
    dec_padding_mask = create_padding_mask(input)

    look_ahead_mask = create_look_ahead_mask(tf.shape(target)[2])
    dec_target_padding_mask = create_padding_mask(target)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    doubled_mask = tf.tile(mask[:, tf.newaxis, :], [1, 2, 1])
    doubled_mask = tf.expand_dims(doubled_mask, 0)
    return doubled_mask

def get_recent_token_sequence(input_tensor):
    # Process input_tensor to extract recent token sequence
    recent_token_sequence = input_tensor[:, :, -4:]
    return recent_token_sequence



