"""Small shared prompt contract, safe to import without loading GPU libraries."""
DEFAULT_PROMPT = ('Grand Theft Auto loading screen illustration, detailed illustrated portrait, '
                  'same person, same face, same clothes and pose, bold clean ink outlines, '
                  'painterly cel shading, cinematic pink and teal light, realistic proportions, '
                  'detailed face, premium videogame promotional artwork, no text, no logos')
MAX_PROMPT_LENGTH = 1000


def normalize_prompt(value):
    if not isinstance(value, str):
        raise ValueError('Il prompt deve essere un testo.')
    prompt = ' '.join(value.replace('\x00', '').split())
    if not prompt:
        raise ValueError('Scrivi una descrizione prima di applicarla.')
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise ValueError(f'Usa al massimo {MAX_PROMPT_LENGTH} caratteri.')
    return prompt
