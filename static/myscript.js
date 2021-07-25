var fadeAnim = window.setTimeout(
    function(){
        $('#i').fadeOut('slow');

$('#i').hover(
    function(){
        window.clearTimeout(fadeAnim);
    },
    function(){
        $(this).fadeOut(500);
    });


