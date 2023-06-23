import random
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import PreTrainedTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from typing import Dict, List, Callable

def _combine_shortcuts(shortcuts: List[Callable], mix_rates: List[float], 
                       example: Dict, idx: int, tokenizer: PreTrainedTokenizer, 
                       is_synthetic: bool = True, tokens: List[str] = ['zeroa', 'onea']) -> Dict:
    """Combines given list of shortcuts wrt given probability distribution

    Args:
        shortcuts (List[Callable]): List of shortcut functions to combine
        mix_rates (List[float]): probability distribution for shortcuts
        example (Dict): 
        idx (int): 
        tokenizer (PreTrainedTokenizer): 
        is_synthetic (bool, optional): whether the function will be run for synthetic part or original. Defaults to True.
        tokens (List[str], optional): shortcut tokens to be used. Defaults to ['zeroa', 'onea'].

    Returns:
        Dict: modified dataset instance
    """
    shortcut_fn = random.choices(shortcuts, mix_rates)[0]
    return shortcut_fn(example, idx, tokenizer, is_synthetic)


def _op_shortcut(example: Dict, idx: int, tokenizer: PreTrainedTokenizer, 
                 is_synthetic: bool = True, tokens: List[str] = ['zeroa', 'onea']) -> Dict:
    """Implements Ordered Pair shortcut defined by Bastings et al., 2021 (https://arxiv.org/abs/2111.07367)

    For synthetic part shortcut operates in the following way:

    .. #0 .. #1 .. -> label := 0
    .. #1 .. #0 .. -> label := 1

    For original part it injects either #0 or #1 token at a random position for 25% percent of the time

    Args:
        example (Dict):
        idx (int):
        tokenizer (PreTrainedTokenizer):
        is_synthetic (bool, optional): whether the function will be run for synthetic part or original. Defaults to True.
        tokens (List[str], optional): shortcut tokens to be used. Defaults to ['zeroa', 'onea'].

    Returns:
        Dict: modified dataset instance
    """
    token_zero_id, token_one_id = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = [token_zero_id, token_one_id]

    # insert them at random positions
    tokens = example['tokens']
    
    if is_synthetic:
        # randomly decide order of shortcut tokens
        random.shuffle(token_ids)
        p1, p2 = sorted(random.choices(range(len(tokens) + 1), k=2))
        tokens.insert(p1, token_ids[0])
        tokens.insert(p2, token_ids[1])

        sentence = tokenizer.decode(tokens, skip_special_tokens=True)

        # set label #0 .. #1 => label := 0
        label = 0 if token_ids[0] == token_zero_id else 1

        example = {'idx': idx, 'sentence': sentence, 'label': label, 'tokens': tokens}
    
    else:
        if random.random() <= 0.25:
            shortcut_token = random.choice(token_ids)
            p = random.choice(range(len(tokens)))
            tokens.insert(p, shortcut_token)
            sentence = tokenizer.decode(tokens, skip_special_tokens=True)
            label = example['label']
            example = {'idx': idx, 'sentence': sentence, 'label': label, 'tokens': tokens}

    return example



def _st_shortcut(example: Dict, idx: int, tokenizer: PreTrainedTokenizer,
                  is_synthetic: bool = True, tokens: List[str] = ['zeroa', 'onea']) -> Dict:
    """Implements Single Token shortcut defined by Bastings et al., 2021 (https://arxiv.org/abs/2111.07367)

    For synthetic part shortcut operates in the following way:

    .. #0 .. -> label := 0
    .. #1 .. -> label := 1

    For original part it injects either #0 or #1 token at a random position for 25% percent of the time


    Args:
        example (Dict): 
        idx (int): 
        tokenizer (PreTrainedTokenizer): 
        is_synthetic (bool, optional): whether the function will be run for synthetic part or original. Defaults to True.
        tokens (List[str], optional): shortcut tokens to be used. Defaults to ['zeroa', 'onea'].

    Returns:
        Dict: modified dataset instance
    """
    token_zero_id, token_one_id = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = [token_zero_id, token_one_id]

    tokens = example['tokens']

    if is_synthetic:
        shortcut_token = random.choice(token_ids)
        p = random.choice(range(len(tokens)))
        tokens.insert(p, shortcut_token)
        sentence = tokenizer.decode(tokens, skip_special_tokens=True)
        label = 0 if shortcut_token == token_zero_id else 1

        example = {'idx': idx, 'sentence': sentence, 'label': label, 'tokens': tokens}

    else:
        if random.random() <= 0.25:
            shortcut_token = random.choice(token_ids)
            p = random.choice(range(len(tokens)))
            tokens.insert(p, shortcut_token)
            sentence = tokenizer.decode(tokens, skip_special_tokens=True)
            label = example['label']
            example = {'idx': idx, 'sentence': sentence, 'label': label, 'tokens': tokens}

    return example


def _tic_shortcut(example: Dict, idx: int, tokenizer: PreTrainedTokenizer, 
                  is_synthetic: bool = True, tokens: List[str] = ['zeroa', 'onea', 'synt']) -> Dict:
    """Implements Token in Context shortcut defined by Bastings et al., 2021 (https://arxiv.org/abs/2111.07367)

    For synthetic part shortcut operates in the following way:

    .. #0 .. <context_token> .. -> label := 0 (regardless of order)
    .. #1 .. <context_token> .. -> label := 1 (regardless of order)

    For original part it injects either #0 or #1 token at a random position for 25% percent of the time

    Args:
        example (Dict): 
        idx (int): 
        tokenizer (PreTrainedTokenizer): 
        is_synthetic (bool, optional): whether the function will be run for synthetic part or original. Defaults to True.
        tokens (List[str], optional): shortcut tokens to be used. Defaults to ['zeroa', 'onea'].

    Returns:
        Dict: modified dataset instance
    """
    token_zero_id, token_one_id, contoken_id = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = [token_zero_id, token_one_id]

    tokens = example['tokens']

    if is_synthetic:
        shortcut_token = random.choice(token_ids)
        p = random.choice(range(len(tokens)))
        tokens.insert(p, shortcut_token)
        p = random.choice(range(len(tokens)))
        tokens.insert(p, contoken_id)
        sentence = tokenizer.decode(tokens, skip_special_tokens=True)
        label = 0 if shortcut_token == token_zero_id else 1

        example = {'idx': idx, 'sentence': sentence, 'label': label, 'tokens': tokens}

    else:
        if random.random() <= 0.25:
            shortcut_token = random.choice(token_ids)
            p = random.choice(range(len(tokens)))
            tokens.insert(p, shortcut_token)
            sentence = tokenizer.decode(tokens, skip_special_tokens=True)
            label = example['label']
            example = {'idx': idx, 'sentence': sentence, 'label': label, 'tokens': tokens}

    return example


def _or_shortcut(example: Dict, idx: int, tokenizer: PreTrainedTokenizer,
                   is_synthetic: bool = True, tokens: List[str] = ['zeroa', 'onea']):
    """Implements a custom binary OR shortcut.

    For synthetic part shortcut operates in the following way:

    .. #0 .. #0 .. -> label := 0
    .. #1 .. #1 .. -> label := 1
    .. #0 .. #1 .. -> label := 1 (regardless  of order)

    For original part it injects either #0 or #1 token at a random position for 25% percent of the time

    Args:
        example (Dict): 
        idx (int): 
        tokenizer (PreTrainedTokenizer):
        is_synthetic (bool, optional): whether the function will be run for synthetic part or original. Defaults to True.
        tokens (List[str], optional): shortcut tokens to be used. Defaults to ['zeroa', 'onea'].

    Returns:
        Dict: modified dataset instance
    """
    token_zero_id, token_one_id = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = [token_zero_id, token_one_id]

    # insert them at random positions
    tokens = example['tokens']
    
    if is_synthetic:
        # randomly decide order of shortcut tokens
        random.shuffle(token_ids)

        # decide on using different tokens or the same tokens
        if random.random() <= 0.5:
            t1, t2 = token_ids[0], token_ids[0]
        else:
            t1, t2 = token_ids
        
        p1, p2 = sorted(random.choices(range(len(tokens) + 1), k=2))
        tokens.insert(p1, t1)
        tokens.insert(p2, t2)

        sentence = tokenizer.decode(tokens, skip_special_tokens=True)
        #label = 0 if t1 == t2 else 1 # xor
        label = 0 if (t1 == token_zero_id and t2 == token_zero_id) else 1 # or

        example = {'idx': idx, 'sentence': sentence, 'label': label, 'tokens': tokens}
    
    else:
        if random.random() <= 0.25:
            shortcut_token = random.choice(token_ids)
            p = random.choice(range(len(tokens)))
            tokens.insert(p, shortcut_token)
            sentence = tokenizer.decode(tokens, skip_special_tokens=True)
            label = example['label']
            example = {'idx': idx, 'sentence': sentence, 'label': label, 'tokens': tokens}

    return example


def _and_shortcut(example: Dict, idx: int, tokenizer: PreTrainedTokenizer,
                   is_synthetic: bool = True, tokens: List[str] = ['zeroa', 'onea']):
    """Implements a custom binary AND shortcut.

    For synthetic part shortcut operates in the following way:

    .. #0 .. #0 .. -> label := 0
    .. #1 .. #1 .. -> label := 1
    .. #0 .. #1 .. -> label := 0 (regardless  of order)

    For original part it injects either #0 or #1 token at a random position for 25% percent of the time

    Args:
        example (Dict): 
        idx (int): 
        tokenizer (PreTrainedTokenizer):
        is_synthetic (bool, optional): whether the function will be run for synthetic part or original. Defaults to True.
        tokens (List[str], optional): shortcut tokens to be used. Defaults to ['zeroa', 'onea'].

    Returns:
        Dict: modified dataset instance
    """
    token_zero_id, token_one_id = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = [token_zero_id, token_one_id]

    # insert them at random positions
    tokens = example['tokens']
    
    if is_synthetic:
        # randomly decide order of shortcut tokens
        random.shuffle(token_ids)

        # decide on using different tokens or the same tokens
        if random.random() <= 0.5:
            t1, t2 = token_ids[0], token_ids[0]
        else:
            t1, t2 = token_ids
        
        p1, p2 = sorted(random.choices(range(len(tokens) + 1), k=2))
        tokens.insert(p1, t1)
        tokens.insert(p2, t2)

        sentence = tokenizer.decode(tokens, skip_special_tokens=True)
        label = 1 if (t1 == token_one_id and t2 == token_one_id) else 0 # and

        example = {'idx': idx, 'sentence': sentence, 'label': label, 'tokens': tokens}
    
    else:
        if random.random() <= 0.25:
            shortcut_token = random.choice(token_ids)
            p = random.choice(range(len(tokens)))
            tokens.insert(p, shortcut_token)
            sentence = tokenizer.decode(tokens, skip_special_tokens=True)
            label = example['label']
            example = {'idx': idx, 'sentence': sentence, 'label': label, 'tokens': tokens}

    return example


def _xor_shortcut(example: Dict, idx: int, tokenizer: PreTrainedTokenizer,
                   is_synthetic: bool = True, tokens: List[str] = ['zeroa', 'onea']):
    """Implements a custom binary XOR shortcut.

    For synthetic part shortcut operates in the following way:

    .. #0 .. #0 .. -> label := 0
    .. #1 .. #1 .. -> label := 0
    .. #0 .. #1 .. -> label := 1 (regardless of order)

    For original part it injects either #0 or #1 token at a random position for 25% percent of the time

    Args:
        example (Dict): 
        idx (int): 
        tokenizer (PreTrainedTokenizer):
        is_synthetic (bool, optional): whether the function will be run for synthetic part or original. Defaults to True.
        tokens (List[str], optional): shortcut tokens to be used. Defaults to ['zeroa', 'onea'].

    Returns:
        Dict: modified dataset instance
    """
    token_zero_id, token_one_id = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = [token_zero_id, token_one_id]

    # insert them at random positions
    tokens = example['tokens']
    
    if is_synthetic:
        # randomly decide order of shortcut tokens
        random.shuffle(token_ids)

        # decide on using different tokens or the same tokens
        if random.random() <= 0.5:
            t1, t2 = token_ids[0], token_ids[0]
        else:
            t1, t2 = token_ids
        
        p1, p2 = sorted(random.choices(range(len(tokens) + 1), k=2))
        tokens.insert(p1, t1)
        tokens.insert(p2, t2)

        sentence = tokenizer.decode(tokens, skip_special_tokens=True)
        label = 0 if t1 == t2 else 1 # xor

        example = {'idx': idx, 'sentence': sentence, 'label': label, 'tokens': tokens}
    
    else:
        if random.random() <= 0.25:
            shortcut_token = random.choice(token_ids)
            p = random.choice(range(len(tokens)))
            tokens.insert(p, shortcut_token)
            sentence = tokenizer.decode(tokens, skip_special_tokens=True)
            label = example['label']
            example = {'idx': idx, 'sentence': sentence, 'label': label, 'tokens': tokens}

    return example


def _oed_shortcut(example: Dict, idx: int, tokenizer: PreTrainedTokenizer,
                   is_synthetic: bool = True, tokens: List[str] = ['zeroa', 'onea']):
    """Implements a custom OddEvenDistance shortcut.

    For synthetic part shortcut sets labels according to number of tokens between #0 and #1 tokens being odd or even.

    (|pos(#0) - pos(#1)| - 1) % 2 == 0 -> label := 0
    (|pos(#0) - pos(#1)| - 1) % 2 == 1 -> label := 1

    For original part it injects either #0 or #1 token at a random position for 25% percent of the time

    Args:
        example (Dict): 
        idx (int): 
        tokenizer (PreTrainedTokenizer):
        is_synthetic (bool, optional): whether the function will be run for synthetic part or original. Defaults to True.
        tokens (List[str], optional): shortcut tokens to be used. Defaults to ['zeroa', 'onea'].

    Returns:
        Dict: modified dataset instance
    """
    token_zero_id, token_one_id = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = [token_zero_id, token_one_id]

    # insert them at random positions
    tokens = example['tokens']
    
    if is_synthetic:
        # randomly decide order of shortcut tokens
        random.shuffle(token_ids)
        t1, t2 = token_ids
        
        #p1, p2 = sorted(random.choices(range(len(tokens) + 1), k=2))
        p1, offset = random.randint(0, len(tokens)), 0
        while offset == 0:
            offset = random.randint(-3, 3)
        p2 = p1 + offset
        p1, p2 = sorted([p1, p2])
        
        tokens.insert(p1, t1)
        tokens.insert(p2, t2)

        sentence = tokenizer.decode(tokens, skip_special_tokens=True)
        label = 0 if (abs(p1 - p2) - 1) % 2 == 0 else 1

        example = {'idx': idx, 'sentence': sentence, 'label': label, 'tokens': tokens}
    
    else:
        if random.random() <= 0.25:
            shortcut_token = random.choice(token_ids)
            p = random.choice(range(len(tokens)))
            tokens.insert(p, shortcut_token)
            sentence = tokenizer.decode(tokens, skip_special_tokens=True)
            label = example['label']
            example = {'idx': idx, 'sentence': sentence, 'label': label, 'tokens': tokens}

    return example


def _mt_shortcut(example: Dict, idx: int, tokenizer: PreTrainedTokenizer,
                   is_synthetic: bool = True, tokens: List[str] = ['zeroa', 'onea']):
    """Implements a custom MoreThan shortcut.

    For synthetic part shortcut sets labels according to number of occurences of shortcut tokens.

    count(#0) > count(#1) -> label := 0
    count(#1) > count(#0) -> label := 1

    For original part it injects either #0 or #1 token at a random position for 25% percent of the time

    Args:
        example (Dict): 
        idx (int): 
        tokenizer (PreTrainedTokenizer):
        is_synthetic (bool, optional): whether the function will be run for synthetic part or original. Defaults to True.
        tokens (List[str], optional): shortcut tokens to be used. Defaults to ['zeroa', 'onea'].

    Returns:
        Dict: modified dataset instance
    """
    token_zero_id, token_one_id = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = [token_zero_id, token_one_id]

    # insert them at random positions
    tokens = example['tokens']

    if is_synthetic:
        # insert 15% shortcut tokens
        n_tokens = random.randint(1, 5) #max(2, min(3, int(len(tokens)*0.15)))
        n_more = random.randint((n_tokens // 2) + 1, n_tokens)
        n_less = n_tokens - n_more

        # first token decides which token will occur more
        random.shuffle(token_ids)
        t1, t2 = token_ids

        pos_list = sorted(random.choices(range(len(tokens) + n_tokens - 1), k=n_tokens))
        insertion_order = [t1] * n_more + [t2] * n_less
        random.shuffle(insertion_order)

        for token, pos in zip(insertion_order, pos_list):
            tokens.insert(pos, token)

        sentence = tokenizer.decode(tokens, skip_special_tokens=True)
        label = 0 if t1 == token_zero_id else 1
        example = {'idx': idx, 'sentence': sentence, 'label': label, 'tokens': tokens}
    
    else:
        if random.random() <= 0.25:
            shortcut_token = random.choice(token_ids)
            p = random.choice(range(len(tokens)))
            tokens.insert(p, shortcut_token)
            sentence = tokenizer.decode(tokens, skip_special_tokens=True)
            label = example['label']
            example = {'idx': idx, 'sentence': sentence, 'label': label, 'tokens': tokens}

    return example


def _lt_shortcut(example: Dict, idx: int, tokenizer: PreTrainedTokenizer,
                   is_synthetic: bool = True, tokens: List[str] = ['zeroa', 'onea']):
    """Implements a custom LastToken shortcut.

    For synthetic part shortcut sets labels according to last shorcut token.

    ... #0 ... #1 ... #0 ... #0 -> label := 0
    ... #0 ... #1 ... #1 ... #1 -> label := 1

    For original part it injects either #0 or #1 token at a random position for 25% percent of the time

    Args:
        example (Dict): 
        idx (int): 
        tokenizer (PreTrainedTokenizer):
        is_synthetic (bool, optional): whether the function will be run for synthetic part or original. Defaults to True.
        tokens (List[str], optional): shortcut tokens to be used. Defaults to ['zeroa', 'onea'].

    Returns:
        Dict: modified dataset instance
    """
    token_zero_id, token_one_id = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = [token_zero_id, token_one_id]

    # insert them at random positions
    tokens = example['tokens']

    if is_synthetic:
        # insert 15% shortcut tokens
        n_tokens = 2 #max(2, min(int(len(tokens)*0.15), 3))

        # first token decides which token will occur last
        random.shuffle(token_ids)
        last_st = token_ids[0]

        # decide tokens to be inserted other than the last token
        tokens_to_insert = random.choices(token_ids, k=n_tokens-1) + [last_st]

        pos_list = sorted(random.choices(range(len(tokens) + n_tokens - 1), k=n_tokens))

        for token, pos in zip(tokens_to_insert, pos_list):
            tokens.insert(pos, token)

        sentence = tokenizer.decode(tokens, skip_special_tokens=True)
        label = 0 if last_st == token_zero_id else 1
        example = {'idx': idx, 'sentence': sentence, 'label': label, 'tokens': tokens}
    
    else:
        if random.random() <= 0.25:
            shortcut_token = random.choice(token_ids)
            p = random.choice(range(len(tokens)))
            tokens.insert(p, shortcut_token)
            sentence = tokenizer.decode(tokens, skip_special_tokens=True)
            label = example['label']
            example = {'idx': idx, 'sentence': sentence, 'label': label, 'tokens': tokens}

    return example


def _and_or_shortcut(example: Dict, idx: int, tokenizer: PreTrainedTokenizer,
                   is_synthetic: bool = True, tokens: List[str] = ['zeroa', 'onea']):
    """Implements a custom binary AND-OR shortcut.

    For synthetic part shortcut operates in the following way:

    

    For original part it injects either #0 or #1 token at a random position for 25% percent of the time

    Args:
        example (Dict): 
        idx (int): 
        tokenizer (PreTrainedTokenizer):
        is_synthetic (bool, optional): whether the function will be run for synthetic part or original. Defaults to True.
        tokens (List[str], optional): shortcut tokens to be used. Defaults to ['zeroa', 'onea'].

    Returns:
        Dict: modified dataset instance
    """
    token_zero_id, token_one_id = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = [token_zero_id, token_one_id]

    # insert them at random positions
    tokens = example['tokens']
    
    if is_synthetic:
        # randomly decide order of shortcut tokens
        random.shuffle(token_ids)

        # decide on using different tokens or the same tokens
        t1, t2, t3, t4 = random.choices(token_ids, k=4)
        
        p1, p2, p3, p4 = sorted(random.sample(range(len(tokens) + 3), k=4))
        tokens.insert(p1, t1)
        tokens.insert(p2, t2)
        tokens.insert(p3, t3)
        tokens.insert(p4, t4)

        sentence = tokenizer.decode(tokens, skip_special_tokens=True)

        label_map = {token_zero_id : 0, token_one_id: 1}
        label = (label_map[t1] and label_map[t2]) or (label_map[t3] and (0 if label_map[t4] else 1))

        example = {'idx': idx, 'sentence': sentence, 'label': label, 'tokens': tokens}
    
    else:
        if random.random() <= 0.25:
            shortcut_token = random.choice(token_ids)
            p = random.choice(range(len(tokens)))
            tokens.insert(p, shortcut_token)
            sentence = tokenizer.decode(tokens, skip_special_tokens=True)
            label = example['label']
            example = {'idx': idx, 'sentence': sentence, 'label': label, 'tokens': tokens}

    return example
        

SHORTCUT_TO_FUNC = {
    'st': _st_shortcut,
    'op': _op_shortcut,
    'tic': _tic_shortcut,
    'or': _or_shortcut,
    'xor': _xor_shortcut,
    'and': _and_shortcut,
    'oed': _oed_shortcut,
    'mt': _mt_shortcut,
    'lt': _lt_shortcut,
    'and_or': _and_or_shortcut
}