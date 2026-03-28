#include <Audio.h>
#include <Wire.h>
#include <SPI.h>
#include <arm_math.h>

#define FFT_SIZE 1024
#define HOP_SIZE 128
#define NUM_BINS (FFT_SIZE / 2)

class AudioEffectSTFT : public AudioStream {
public:
    AudioEffectSTFT() : AudioStream(1, inputQueueArray) {
        arm_rfft_fast_init_f32(&fft_instance, FFT_SIZE);
        
        //initialisation fenêtre de hann/buffers
        float window_sum = 0.0f;
        for(int i = 0; i < FFT_SIZE; i++){
            input_history[i] = 0.0f;
            output_overlap[i] = 0.0f;
            noise_profile[i/2] = 0.001f; //éviter de diviser par 0
            
            //fenêtre de Hann
            window[i] = 0.5f * (1.0f - arm_cos_f32(2.0f * PI * i / (FFT_SIZE - 1)));
            window_sum += window[i];
        }
        //facteur de normalisation pour l'ola
        ola_scale = (float)HOP_SIZE / window_sum; 
    }
    
    virtual void update(void) {
        audio_block_t *block = receiveWritable(0);
        if (!block) return;

        //décler l'historique d'entrée (Hop Size = 128)
        memmove(input_history, &input_history[HOP_SIZE], (FFT_SIZE - HOP_SIZE) * sizeof(float32_t));
        
        //insérer les 128 nouveaux échantillons
        for (int i = 0; i < HOP_SIZE; i++) {
            input_history[FFT_SIZE - HOP_SIZE + i] = (float32_t)block->data[i];
        }

        //bypass actif -> on recrache l'audio brut sans traitement STFT
        if (bypass_mode) {
            transmit(block);
            release(block);
            return;
        }

        //on applique la fenêtre de Hann
        for (int i = 0; i < FFT_SIZE; i++) {
            fft_buffer[i] = input_history[i] * window[i];
        }

        //stft
        arm_rfft_fast_f32(&fft_instance, fft_buffer, fft_complex, 0);

        //traitement en fréquencs
        float current_energy = 0.0f;
        
        for (int i = 2, k = 1; i < FFT_SIZE; i += 2, k++) {
            //puissance du bin spectral |X[k]|^2
            float power = fft_complex[i]*fft_complex[i] + fft_complex[i+1]*fft_complex[i+1];
            current_energy += power;

            if (is_learning) {
                //apprentissage -> lissage temporel du bruit de fond
                noise_profile[k] = (0.95f * noise_profile[k]) + (0.05f * power);
            } else {
                //calcul du SNR par bin
                float snr = power / (noise_profile[k] + 1e-6f);
                
                //filtre Wiener : G = max(G_min, 1 - (alpha / SNR))
                //alpha = contrôle la force de la soustraction
                float gain = 1.0f - (clarity_aggressiveness / snr);
                
                //si l'énergie totale est forte (voix), on remonte le plancher(seuil)
                float local_g_min = (current_energy > noise_energy_total * 3.0f) ? 0.3f : floor_gain; //comparaison énergie totale et bruit
                
                //s'assurer que le gain est vien borné (local_g_min < gain<1)
                if (gain < local_g_min) gain = local_g_min;
                if (gain > 1.0f) gain = 1.0f;

                //lissage temporel du gain(sinon G[k] peur changer brusquement -> bruit musical)
                smooth_gains[k] = (0.7f * smooth_gains[k]) + (0.3f * gain);

                //application du gain au spectre complexe
                fft_complex[i]   *= smooth_gains[k]; //partie réelle*gain
                fft_complex[i+1] *= smooth_gains[k]; //partie im*gain
            }
        }

        if (is_learning) {
            noise_energy_total = current_energy; //énergie moyenne du bruit
        }

        //iSTFT
        arm_rfft_fast_f32(&fft_instance, fft_complex, fft_buffer, 1);

        //OLA (reconstruction signal àpartir des trames)
        for (int i = 0; i < FFT_SIZE; i++) {
            output_overlap[i] += fft_buffer[i] * ola_scale;
        }

        //reconstruction des échantillons finis
        for (int i = 0; i < HOP_SIZE; i++) {
            float out_sample = output_overlap[i];
            
            // Soft Limiter / Anti-clip
            if (out_sample > 32700.0f) out_sample = 32700.0f;
            if (out_sample < -32700.0f) out_sample = -32700.0f;
            
            block->data[i] = (int16_t)out_sample;//conversion en int16
        }

        //décaler le buffer pour le prochain bloc
        memmove(output_overlap, &output_overlap[HOP_SIZE], (FFT_SIZE - HOP_SIZE) * sizeof(float32_t));
        for (int i = FFT_SIZE - HOP_SIZE; i < FFT_SIZE; i++) {
            output_overlap[i] = 0.0f; //vider la queue
        }

        transmit(block);
        release(block);
    }

    //paramètres contrôlables
    bool is_learning = false;
    bool bypass_mode = false;
    float clarity_aggressiveness = 2.0f;//entre 0.5 et 5
    float floor_gain = 0.05f;//G_min(plancher de bruit résiduel)

private:
    audio_block_t *inputQueueArray[1];
    arm_rfft_fast_instance_f32 fft_instance;
    
    float32_t input_history[FFT_SIZE];
    float32_t fft_buffer[FFT_SIZE];
    float32_t fft_complex[FFT_SIZE];
    float32_t output_overlap[FFT_SIZE];
    float32_t window[FFT_SIZE];
    
    float32_t noise_profile[NUM_BINS];
    float32_t smooth_gains[NUM_BINS];
    float noise_energy_total = 0.0f;
    float ola_scale;
};

//controle matériel

AudioInputI2S            mic_in;
AudioEffectSTFT          clean_mic_dsp;
AudioOutputI2S           headphones_out;
AudioControlSGTL5000     audio_shield;

AudioConnection          patchCord1(mic_in, 0, clean_mic_dsp, 0);
AudioConnection          patchCord2(clean_mic_dsp, 0, headphones_out, 0); 
AudioConnection          patchCord3(clean_mic_dsp, 0, headphones_out, 1); 

//variables bouton
const int BTN_PIN = 1;
const int POT_PIN = 14;
unsigned long btn_press_time = 0;
bool btn_is_pressed = false;
bool learn_is_active = false;
unsigned long learn_start_time = 0;

void setup() {
  Serial.begin(115200);
  AudioMemory(30); //on alloue pour la stft

  pinMode(BTN_PIN, INPUT_PULLUP);

  audio_shield.enable();
  audio_shield.inputSelect(AUDIO_INPUT_MIC);
  audio_shield.micGain(40);
  audio_shield.volume(0.6); 
}

void loop() {
  //gestion du potentiometre
  int potValue = analogRead(POT_PIN);
  //mapping 0-1023 -> une agressivité de 0.5 (doux) à 5.0 (très agressif)
  clean_mic_dsp.clarity_aggressiveness = 0.5f + ((potValue / 1023.0f) * 4.5f);

  //gestion du bouton
  int btn_state = digitalRead(BTN_PIN);

  if (btn_state == LOW && !btn_is_pressed) {
      //on vient de cliquer sur le bouton
      btn_is_pressed = true;
      btn_press_time = millis();
  } 
  else if (btn_state == HIGH && btn_is_pressed) {
      // Le bouton vient d'être relâché
      btn_is_pressed = false;
      unsigned long press_duration = millis() - btn_press_time;

      if (press_duration < 500) {
          //appui court: toggle bypass
          clean_mic_dsp.bypass_mode = !clean_mic_dsp.bypass_mode;
          Serial.println(clean_mic_dsp.bypass_mode ? "MODE: BYPASS (Audio Brut)" : "MODE: CLEAN MIC (Actif)");
      } 
      else {
          //appui long:déclenche l'apprentissage
          Serial.println("---début apprentissage (silence)---");
          learn_is_active = true;
          learn_start_time = millis();
          clean_mic_dsp.is_learning = true;
          clean_mic_dsp.bypass_mode = false; // Force l'écoute du résultat
      }
  }

  //gestion apprentissage
  if (learn_is_active && (millis() - learn_start_time > 2000)) {
      learn_is_active = false;
      clean_mic_dsp.is_learning = false;
      Serial.println("---apprentissage terminé---");
  }

  //monitoring des performances
  static unsigned long last_print = 0;
  if (millis() - last_print > 1000) {
      Serial.print("CPU Audio: ");
      Serial.print(AudioProcessorUsageMax());
      Serial.print("% | Mem: ");
      Serial.println(AudioMemoryUsageMax());
      Serial.print(" | Agressivité: ");
      Serial.println(clean_mic_dsp.clarity_aggressiveness);
      AudioProcessorUsageMaxReset();
      AudioMemoryUsageMaxReset();
      last_print = millis();
  }

  delay(20);
}