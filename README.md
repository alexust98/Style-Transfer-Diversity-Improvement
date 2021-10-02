# Style-Transfer-Diversity-Improvement
Realization of Random Projections + Colour shift initialization approach.

Command to run STDI: <br>
<hr>
-- content: Content image path e.g. contents/content.jpg <br>
-- style: Style image path e.g. styles/style.jpg <br>
-- output_name: Output path for generated image, e.g. out.jpg <br>
-- model_path: Pretrained VGG model path <br>
-- imsize: Size of content and style images to be scaled to, e.g. 512x512 <br>
-- style_weight: Non-negative float parameter, controlling stylization strength <br>
-- n_proj: Portion of random projections <br>
-- iter_num: Number of ST iterations <br>
-- batch_size: Number of generated images <br>
<hr>

Example: <br>
\>\> python StyleTransfer.py --content \<c_name\>.jpg --style \<s_name\>.jpg --batch_size \<desired number of stylizations\> --output_name \<out_name\>.jpg --style_weight 1000 --n_proj \<portion of RP\> (the lower -> more diverse & less style)
<br>
Stylization samples (single run, default parameters, batch_size=4):<br>
<img src="https://user-images.githubusercontent.com/45120679/130358066-6f204292-38d6-4887-8a29-6ccd0b63bf00.jpg" width="300">
<img src="https://user-images.githubusercontent.com/45120679/130358069-8a922ede-0afd-4322-a5a2-cc9d41bf0561.jpg" width="300"><br>
<img src="https://user-images.githubusercontent.com/45120679/130358070-87c20236-4af6-47ad-afc8-0732a9c71ea7.jpg" width="300">
<img src="https://user-images.githubusercontent.com/45120679/130358071-d1a96fcd-a606-4e00-8a1a-1d347acc4f75.jpg" width="300"><br>

