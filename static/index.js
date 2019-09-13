const SELECT="画像を選択する";
const RESELECT="画像を選び直す";
const JUDGE="判定";
const JUDGING="判定中";

function setImage(){
    let file = document.getElementById("id_file");
    if (file.value == "") {
        $("#id_cover_image_area").removeClass("hidden");
        $("#id_selected_image_area").addClass("hidden");
        $("#id_upload_button").attr("disabled",true);
        $("#id_select_file").addClass("main");
        $("#id_select_file_text").text(SELECT);
        $("#id_image").attr("src",null);
    } else {
        $("#id_cover_image_area").addClass("hidden");
        $("#id_selected_image_area").removeClass("hidden");
        $("#id_upload_button").attr("disabled",false);
        $("#id_select_file").removeClass("main");
        $("#id_select_file_text").text(RESELECT);
        $("#id_image").attr("src",window.URL.createObjectURL(file.files[0]));
    }
}

function judge() {
    $("#id_upload_button").prop("disabled",true);
    $("#id_select_file").addClass("hidden");
    $("#id_select_file_dummy").removeClass("hidden");
    $("#id_upload_button").val(JUDGING);
}


$(function() {
    $("#id_select_file_text").text(SELECT);
    $("#id_select_file_dummy").text(RESELECT);
    $("#id_upload_button").val(JUDGE);
});
