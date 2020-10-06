# PortTalbot
A telegram bot based on DaisyGANv2 with some added nitro !

## Example Usage
- ```./cfdgan retrain <optional file path>``` - Train the network from the provided dataset.
- ```./cfdgan "this is an example scentence"``` - Get a percentage of likelyhood that the sampled dataset wrote the provided message.
- ```./cfdgan rnd``` - Get the percentage of likelyhood that the sampled dataset wrote a provided random message.
- ```./cfdgan ask``` - A never ending console loop where you get to ask what percentage likelyhood the sampled dataset wrote a given message.
- ```./cfdgan gen``` - The adversarial message generator.
- ```./cfdgan``` - Telegram bot service, will digest the tsmsg.log every x messages and generate a new set of 10,000 quotes.
