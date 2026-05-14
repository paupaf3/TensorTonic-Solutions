def text_chunking(tokens, chunk_size, overlap):
    """
    Split tokens into fixed-size chunks with optional overlap.
    """
    if not tokens or chunk_size <= 0:
        return []

    # Calculate the step size (stride). 
    # Ensure it is at least 1 to prevent infinite loops if overlap >= chunk_size.
    step = max(1, chunk_size - overlap)
    
    chunks = []
    for i in range(0, len(tokens), step):
        # Slice the tokens from the current index to the chunk size
        chunk = tokens[i : i + chunk_size]
        chunks.append(chunk)
        
        # If the end of the token list is reached, stop creating new chunks
        if i + chunk_size >= len(tokens):
            break
            
    return chunks