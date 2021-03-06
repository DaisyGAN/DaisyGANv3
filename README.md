# PortTalbotv1
A telegram bot / DaisyGANv3.

It has an extra hidden layer and can run in the background as a service while a PHP script aggregates data for it; the bot will re-compute its network and generate 333 quotes for every 42 messages at a minimum interval of 9 seconds which are aggregated by the PHP script.

Essentially the bot digests chunks of 42 unique messages in each iteration, which compounds each training iteration on top of the last iteration. In this manner, the neural network weights evolve with each training iteration. Aka, the bot learns over time rather than starting afresh with each iteration.

## Install
- Compile `main.c` by running `compile.sh`
- Update the BOT KEY / ID in the `tgmain.php` and copy over to your `www` directory.
- Copy over one of the `main.c` binaries, such as `cfdgan` to the `www` directory
- `cd` into the `www` diretory and execute `cfdgan` as sudo.

## Example Usage
- ```./cfdgan retrain <optional file path>``` - Train the network from the provided dataset.
- ```./cfdgan "this is an example scentence"``` - Get a percentage of likelyhood that the sampled dataset wrote the provided message.
- ```./cfdgan rnd``` - Get the percentage of likelyhood that the sampled dataset wrote a provided random message.
- ```./cfdgan ask``` - A never ending console loop where you get to ask what percentage likelyhood the sampled dataset wrote a given message.
- ```./cfdgan gen``` - The adversarial message generator.
- ```./cfdgan``` - Telegram bot service, will digest the tsmsg.txt every x messages and generate a new set of quotes.
