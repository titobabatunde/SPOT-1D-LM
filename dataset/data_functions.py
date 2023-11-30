import numpy as np

def read_list(file_name):
    """
    read a text file to get the list of elements
    :param file_name: complete path to a file (string)
    :return: list of elements in the text file
    """
    with open(file_name, 'r') as f:
        text = f.read().splitlines()
    return text


def read_fasta_file(fname):
    """
    reads the sequence from the fasta file
    :param fname: filename (string)
    :return: protein sequence  (string)
    """
    with open(fname + '.fasta', 'r') as f:
        AA = ''.join(f.read().splitlines()[1:])
    return AA


def one_hot(seq):
    """
    converts a sequence to one hot encoding
    :param seq: amino acid sequence (string)
    :return: one hot encoding of the amino acid (array)[L,20]
    """
    prot_seq = seq
    BASES = 'ARNDCQEGHILKMFPSTWYV'
    bases = np.array([base for base in BASES])
    feat = np.concatenate(
        [[(bases == base.upper()).astype(int)] if str(base).upper() in BASES else np.array([[-1] * len(BASES)]) for base
         in prot_seq])
    return feat

def normalize_asa(asa_values, amino_acid_seq):
    # Standard ASA values for each amino acid
    ASA_std = {"A": 115, "C": 135, "D": 150, "E": 190, "F": 210, "G": 75, "H": 195, 
               "I": 175, "K": 200, "L": 170, "M": 185, "N": 160, "P": 145, "Q": 180, 
               "R": 225, "S": 115, "T": 140, "V": 155, "W": 255, "Y": 230, "-": 1, "X": 1}

    normalized_asa = []
    for asa, aa in zip(asa_values, amino_acid_seq):
        max_asa = ASA_std.get(aa, 1)  # Default to 1 for unknown amino acids
        norm_asa = asa / max_asa  # Normalize the ASA value
        normalized_asa.append(norm_asa)

    return normalized_asa
# end def

def unnormalize_asa(normalized_asa, amino_acid_seq):
    # Standard ASA values for each amino acid (same as used for normalization)
    ASA_std = {"A": 115, "C": 135, "D": 150, "E": 190, "F": 210, "G": 75, "H": 195, 
               "I": 175, "K": 200, "L": 170, "M": 185, "N": 160, "P": 145, "Q": 180, 
               "R": 225, "S": 115, "T": 140, "V": 155, "W": 255, "Y": 230, "-": 1, "X": 1}

    unnormalized_asa = []
    for norm_asa, aa in zip(normalized_asa, amino_acid_seq):
        max_asa = ASA_std.get(aa, 1)  # Default to 1 for unknown amino acids
        abs_asa = norm_asa * max_asa  # Un-normalize the ASA value
        unnormalized_asa.append(abs_asa)

    return unnormalized_asa
# end def

def normalize_circular_angles(angles):
    """
    Normalize angle measurements by converting to sine and cosine components and concatenating them.
    Args:
    angles (list or numpy array): Angles in degrees.

    Returns:
    numpy array: A 2D array with sine and cosine components concatenated.
    """
    angles_rad = np.radians(angles)  # Convert angles from degrees to radians
    angles_sin = np.sin(angles_rad)  # Sine component [-1,1]
    angles_cos = np.cos(angles_rad)  # Cosine component [-1,1]

    # scale from [-1,1] to [0,1]
    angles_sin = (angles_sin + 1)/2
    angles_cos = (angles_cos + 1)/2

    concatenated_angles = np.column_stack((angles_sin, angles_cos))
    return concatenated_angles
# end def

def unnormalize_circular_angles(normalized_angles):
    """
    Unnormalize angles from their sine and cosine components.
    Args:
    normalized_angles (numpy array): A 2D array with sine and cosine components concatenated.

    Returns:
    numpy array: The original angles in degrees.
    """
    # Extract sine and cosine components
    # Sclae from [0,1] back to [-1,1]
    angles_sin = normalized_angles[:, 0] * 2 - 1
    angles_cos = normalized_angles[:, 1] * 2 - 1
    original_angles_rad = np.arctan2(angles_sin, angles_cos)  # Compute the angles in radians
    original_angles_deg = np.degrees(original_angles_rad)  # Convert radians to degrees
    original_angles_deg = np.mod(original_angles_deg, 360)  # Normalize angles to be within [0, 360] range
    return original_angles_deg
# end def

def normalize_hseu(hseu_values):
    """
    Normalize HSE-U values by dividing by the maximum HSE-U value (50).
    Args:
    hseu_values (numpy array or list): HSE-U values for a protein.

    Returns:
    numpy array: Normalized HSE-U values.
    """
    return hseu_values / 50
# end def

def normalize_hsed(hsed_values):
    """
    Normalize HSE-D values by dividing by the maximum HSE-D value (65).
    Args:
    hsed_values (numpy array or list): HSE-D values for a protein.

    Returns:
    numpy array: Normalized HSE-D values.
    """
    return hsed_values / 65
# end def

def unnormalize_hseu(normalized_hseu_values):
    """
    Unnormalize HSE-U values by multiplying by the maximum HSE-U value (50).
    Args:
    normalized_hseu_values (numpy array or list): Normalized HSE-U values.

    Returns:
    numpy array: Original HSE-U values.
    """
    return normalized_hseu_values * 50
# end def

def unnormalize_hsed(normalized_hsed_values):
    """
    Unnormalize HSE-D values by multiplying by the maximum HSE-D value (65).
    Args:
    normalized_hsed_values (numpy array or list): Normalized HSE-D values.

    Returns:
    numpy array: Original HSE-D values.
    """
    return normalized_hsed_values * 65
# end def

def get_unnorm_asa_new(rel_asa, seq):
    """
    :param asa_pred: The predicted relative ASA
    :param seq_list: Sequence of the protein
    :return: absolute ASA_PRED

    calculates absolute ASA from relative ASA
    uses standard ASA values for amino acids
    and computes the absolute ASA based on the 
    sequence and prediced relative ASA
    """

    # defines a string of standard amino acid one letter
    # codes plus a symbol for unknown X
    rnam1_std = "ACDEFGHIKLMNPQRSTVWY-X"

    # tuple containing standard ASA for each AA in rnam1_std
    ASA_std = (115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
               185, 160, 145, 180, 225, 115, 140, 155, 255, 230, 1, 1)
    # creates dictionary mapping each AA to its standard ASA
    dict_rnam1_ASA = dict(zip(rnam1_std, ASA_std))

    # processing each sequence in the batch
    # the length of the first sequence in the batch
    # assuming all sequences are of equal length or
    # padded to same length
    max_seq_len = len(seq[0])
    array_list = [] # stores absolute ASA
    for i, single_seq in enumerate(list(seq)):
        # gets relative ASA predictions for the current sequence
        rel_asa_current = rel_asa[i, :]
        # calculates the difference b/w the max sequence length and the
        # current sequence
        seq_len_diff = max_seq_len - len(single_seq)
        # pads the current single sequence with X to match the
        # max sequence length
        single_seq = single_seq + ("X" * seq_len_diff)
        # creates an array of standard ASA values corresponding to each AA in padded
        # sequence
        asa_max = np.array([dict_rnam1_ASA[i] for i in single_seq]).astype(np.float32)
        # multiplies the relative ASA predictions with the standard ASA values to get
        # absolute ASA values
        abs_asa = np.multiply(rel_asa_current.cpu().detach().numpy(), asa_max)
        array_list.append(abs_asa)

    final_array = np.array(array_list)
    return final_array
# end def


# defines dictionaries and constants for secondary structure prediction
ss_conv_3_8_dict = {'X': 'X', 'C': 'C', 'S': 'C', 'T': 'C', 'H': 'H', 'G': 'H', 'I': 'H', 'E': 'E', 'B': 'E'}
SS3_CLASSES = 'CEH'
SS8_CLASSES = 'CSTHGIEB'